# -*- coding: utf-8 -*-
"""
main_qm9.py

اسکریپت اصلی آموزش و ارزیابی PAMNet روی دیتاست QM9.

این فایل کارهای زیر را انجام می‌دهد:
1. آرگومان‌های خط فرمان را می‌خواند.
2. دیتاست QM9 را بارگذاری و به train/validation/test تقسیم می‌کند.
3. یکی از مدل‌های PAMNet یا PAMNet_s را با یکی از چهار حالت معماری می‌سازد:
   - original: معماری اصلی بدون weight sharing و بدون basis reduction
   - ws: فقط weight sharing
   - br: فقط کاهش تعداد basis functions
   - wsbr: هر دو تغییر weight sharing و basis reduction
4. مدل را آموزش می‌دهد.
5. بهترین مدل را براساس کمترین خطای validation ذخیره می‌کند.
6. خروجی‌های آزمایش را در فایل‌های CSV و JSON ذخیره می‌کند.

نکته‌ی مهم برای مقایسه با کد اولیه:
در نسخه‌ی اولیه، subset دیتاست ثابت 5000 نمونه بود و تغییرات مدل داخل کد hard-code شده بود.
در این نسخه، همان ایده‌ها به شکل آرگومان خط فرمان قابل کنترل شده‌اند تا بتوان چند آزمایش را منظم‌تر اجرا و مقایسه کرد.
"""

# -----------------------------
# کتابخانه‌های استاندارد پایتون
# -----------------------------
import argparse  # خواندن ورودی‌های خط فرمان، مثل --epochs و --variant
import csv       # ذخیره‌ی نتایج هر epoch و خلاصه‌ی نهایی در CSV
import json      # ذخیره‌ی تنظیمات اجرا در فایل JSON
import os        # ساخت پوشه‌ها و کار با مسیرها
import os.path as osp  # نسخه‌ی کوتاه‌تر os.path برای ساخت مسیر فایل‌ها
import random    # کنترل seed مربوط به random پایتون
import time      # اندازه‌گیری زمان آموزش و inference
from datetime import datetime  # ساخت timestamp برای نام‌گذاری runها

# -----------------------------
# کتابخانه‌های عددی و deep learning
# -----------------------------
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

# DataLoader جدید PyTorch Geometric؛ نسخه‌ی قدیمی torch_geometric.data.DataLoader deprecated شده است.
from torch_geometric.loader import DataLoader

# tqdm فقط برای نمایش progress bar هنگام آموزش استفاده می‌شود.
from tqdm.auto import tqdm

# Scheduler گرم‌کننده‌ی learning rate در ابتدای آموزش.
from warmup_scheduler import GradualWarmupScheduler

# کلاس دیتاست QM9، مدل‌ها و EMA از فایل‌های پروژه وارد می‌شوند.
from datasets import QM9
from models import PAMNet, PAMNet_s, Config
from utils import EMA


# نگاشت شماره‌ی target در QM9 به اسم ویژگی شیمیایی.
# این دیکشنری فقط برای خواناتر شدن log و خروجی‌ها استفاده می‌شود.
TARGET_NAMES = {
    0: "mu",       # Dipole moment
    1: "alpha",    # Isotropic polarizability
    2: "HOMO",     # Highest occupied molecular orbital energy
    3: "LUMO",     # Lowest unoccupied molecular orbital energy
    4: "gap",      # HOMO-LUMO gap
    5: "R2",       # Electronic spatial extent
    6: "ZPVE",     # Zero point vibrational energy
    7: "U0",       # Internal energy at 0K
    8: "U",        # Internal energy at 298.15K
    9: "H",        # Enthalpy at 298.15K
    10: "G",       # Free energy at 298.15K
    11: "Cv",      # Heat capacity
    12: "U0_atom", # Atomization energy corresponding to U0
    13: "U_atom",  # Atomization energy corresponding to U
    14: "H_atom",  # Atomization energy corresponding to H
    15: "G_atom",  # Atomization energy corresponding to G
    16: "A",       # Rotational constant A
    17: "B",       # Rotational constant B
    18: "C",       # Rotational constant C
}


def set_seed(seed: int) -> None:
    """
    ثابت کردن seed برای تکرارپذیری آزمایش.

    وقتی seed ثابت باشد، تا حد ممکن split، shuffle و مقداردهی اولیه‌ی مدل در اجراهای مختلف یکسان می‌ماند.
    deterministic=True باعث می‌شود cuDNN الگوریتم‌های deterministic را ترجیح دهد.
    benchmark=False مانع انتخاب خودکار سریع‌ترین الگوریتم cuDNN می‌شود چون آن انتخاب ممکن است بین اجراها متفاوت باشد.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """
    تعداد پارامترهای قابل‌آموزش مدل را می‌شمارد.

    فقط پارامترهایی حساب می‌شوند که requires_grad=True دارند؛ یعنی optimizer آن‌ها را update می‌کند.
    این عدد برای مقایسه‌ی معماری original با weight sharing و basis reduction مهم است.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_qm9_target(cli_target: int, use_atom_mapping: bool = True) -> int:
    """
    target واردشده در command line را به ستون واقعی data.y در QM9 تبدیل می‌کند.

    در کد اصلی PAMNet، targetهای 7 تا 10 به ستون‌های 12 تا 15 منتقل می‌شدند.
    دلیلش این است که برای انرژی‌ها معمولاً atomization energy استفاده می‌شد:
        7  -> 12 یعنی U0_atom
        8  -> 13 یعنی U_atom
        9  -> 14 یعنی H_atom
        10 -> 15 یعنی G_atom

    اگر --no_atom_mapping داده شود، همین mapping خاموش می‌شود و همان شماره‌ی ورودی استفاده می‌شود.
    """
    if use_atom_mapping and cli_target in [7, 8, 9, 10]:
        return cli_target + 5
    return cli_target


@torch.no_grad()
def evaluate(model, loader, ema, device):
    """
    محاسبه‌ی MAE مدل روی validation یا test.

    - torch.no_grad باعث می‌شود gradient محاسبه نشود؛ پس evaluation سریع‌تر و کم‌مصرف‌تر است.
    - model.eval مدل را در حالت ارزیابی قرار می‌دهد.
    - ema.assign وزن‌های EMA را موقتاً روی مدل می‌گذارد.
    - ema.resume بعد از ارزیابی وزن‌های اصلی آموزش را برمی‌گرداند.
    """
    mae = 0.0
    model.eval()

    # استفاده از وزن‌های میانگین متحرک نمایی برای ارزیابی پایدارتر.
    ema.assign(model)

    for data in loader:
        # انتقال batch به CPU یا GPU انتخاب‌شده.
        data = data.to(device)

        # خروجی مدل یک مقدار پیش‌بینی‌شده برای هر molecule در batch است.
        output = model(data)

        # جمع absolute error برای همه‌ی graphهای این batch.
        mae += (output - data.y).abs().sum().item()

    # بازگرداندن وزن‌های اصلی مدل بعد از evaluation.
    ema.resume(model)

    # میانگین خطای مطلق روی کل دیتاست.
    return mae / len(loader.dataset)


@torch.no_grad()
def evaluate_with_time(model, loader, ema, device):
    """
    علاوه بر MAE، زمان inference روی test loader را هم اندازه می‌گیرد.

    اگر GPU استفاده شود، قبل و بعد از اندازه‌گیری torch.cuda.synchronize صدا زده می‌شود.
    چون عملیات CUDA asynchronous هستند و بدون synchronize زمان اندازه‌گیری‌شده دقیق نیست.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.time()
    mae = evaluate(model, loader, ema, device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.time() - start
    return mae, elapsed


def write_csv_row(path, row):
    """
    یک ردیف را به فایل CSV اضافه می‌کند.

    اگر فایل CSV هنوز وجود نداشته باشد، ابتدا header با نام ستون‌ها نوشته می‌شود.
    از این تابع برای ذخیره‌ی log هر epoch و همچنین summary نهایی استفاده می‌شود.
    """
    # مطمئن می‌شویم پوشه‌ی خروجی وجود دارد.
    os.makedirs(osp.dirname(path), exist_ok=True)

    # اگر فایل از قبل وجود داشته باشد، دیگر header نمی‌نویسیم.
    file_exists = osp.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    """
    نقطه‌ی اصلی اجرای برنامه.

    همه‌ی مراحل experiment از این تابع شروع می‌شود:
    خواندن آرگومان‌ها، ساخت دیتاست، ساخت مدل، آموزش، evaluation، ذخیره‌ی مدل و ذخیره‌ی نتایج.
    """
    parser = argparse.ArgumentParser()

    # -----------------------------
    # تنظیمات عمومی اجرا
    # -----------------------------
    parser.add_argument("--gpu", type=int, default=0, help="GPU number.")
    parser.add_argument("--seed", type=int, default=480, help="Random seed.")
    parser.add_argument("--dataset", type=str, default="QM9", help="Dataset to be used.")
    parser.add_argument("--model", type=str, default="PAMNet", choices=["PAMNet", "PAMNet_s"], help="Model.")

    # variant مشخص می‌کند کدام تغییر معماری فعال باشد.
    # original: حالت اصلی
    # ws: فقط weight sharing
    # br: فقط basis reduction
    # wsbr: ترکیب weight sharing و basis reduction
    parser.add_argument(
        "--variant",
        type=str,
        default="original",
        choices=["original", "ws", "br", "wsbr"],
        help="Architecture variant: original, ws, br, wsbr.",
    )

    # -----------------------------
    # hyperparameterهای آموزش
    # -----------------------------
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)

    # target همان index ویژگی موردنظر برای پیش‌بینی در QM9 است.
    # به صورت پیش‌فرض target=7 داده شده که اگر atom mapping فعال باشد به U0_atom تبدیل می‌شود.
    parser.add_argument(
        "--target",
        type=int,
        default=6,
        help="CLI target index. Default 6 maps to ZPVE.",
    )
    parser.add_argument(
        "--no_atom_mapping",
        action="store_true",
        help="Disable original 7..10 -> 12..15 atomization-energy target mapping.",
    )

    # cutoff_l برای graph محلی و cutoff_g برای graph سراسری استفاده می‌شود.
    parser.add_argument("--cutoff_l", type=float, default=5.0)
    parser.add_argument("--cutoff_g", type=float, default=5.0)

    # -----------------------------
    # اندازه‌ی split دیتاست
    # -----------------------------
    # در کد اولیه، QM9-5k به صورت ثابت 4000/500/500 بود.
    # اینجا همان default حفظ شده ولی قابل تغییر با command line شده است.
    parser.add_argument("--train_size", type=int, default=4000)
    parser.add_argument("--val_size", type=int, default=500)
    parser.add_argument("--test_size", type=int, default=500)

    # -----------------------------
    # تنظیمات basis functionها
    # -----------------------------
    # حالت original: SBF=7x6 و RBF=16
    parser.add_argument("--num_spherical", type=int, default=7)
    parser.add_argument("--num_radial", type=int, default=6)
    parser.add_argument("--num_rbf", type=int, default=16)

    # حالت basis reduction: SBF=3x4 و RBF=8
    parser.add_argument("--reduced_num_spherical", type=int, default=3)
    parser.add_argument("--reduced_num_radial", type=int, default=4)
    parser.add_argument("--reduced_num_rbf", type=int, default=8)

    # -----------------------------
    # تنظیمات خروجی و لاگ
    # -----------------------------
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--disable_tqdm", action="store_true")
    args = parser.parse_args()

    # تبدیل مقدار variant به دو flag واضح برای مدل.
    use_weight_sharing = args.variant in ["ws", "wsbr"]
    use_basis_reduction = args.variant in ["br", "wsbr"]

    # اگر --no_atom_mapping داده نشده باشد، mapping اصلی PAMNet فعال است.
    use_atom_mapping = not args.no_atom_mapping
    actual_target = resolve_qm9_target(args.target, use_atom_mapping=use_atom_mapping)

    # جلوگیری از انتخاب target نامعتبر.
    if actual_target < 0 or actual_target > 18:
        raise ValueError(f"Invalid resolved QM9 target index: {actual_target}. It must be between 0 and 18.")

    # انتخاب device؛ اگر GPU در دسترس باشد از CUDA استفاده می‌شود، وگرنه CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

        # reset_peak_memory_stats باعث می‌شود peak memory فقط مربوط به همین run باشد.
        torch.cuda.reset_peak_memory_stats(device)

    # ثابت کردن seed بعد از انتخاب device.
    set_seed(args.seed)

    class MyTransform(object):
        """
        transform مربوط به دیتاست QM9.

        QM9 در data.y چندین target دارد. این transform فقط ستون target انتخاب‌شده را نگه می‌دارد
        تا مدل دقیقاً برای همان ویژگی آموزش ببیند.
        """
        def __call__(self, data):
            data.y = data.y[:, actual_target]
            return data

    # مسیر ذخیره/خواندن دیتاست. اگر دیتاست وجود نداشته باشد، کلاس QM9 معمولاً آن را دانلود/پردازش می‌کند.
    path = osp.join(".", "data", args.dataset)
    dataset = QM9(path, transform=MyTransform()).shuffle()

    # تعداد کل نمونه‌های موردنیاز برای train + validation + test.
    total_size = args.train_size + args.val_size + args.test_size

    # چک امنیتی برای اینکه بیشتر از حجم واقعی دیتاست نمونه نخواهیم.
    if total_size > len(dataset):
        raise ValueError(f"Requested split size {total_size} exceeds dataset size {len(dataset)}.")

    # ابتدا subset موردنیاز برداشته می‌شود، سپس split انجام می‌شود.
    # default دقیقاً همان 5000 نمونه‌ی کد ساده است: 4000 train، 500 validation، 500 test.
    subset = dataset[:total_size]
    train_dataset = subset[:args.train_size]
    val_dataset = subset[args.train_size:args.train_size + args.val_size]
    test_dataset = subset[args.train_size + args.val_size:total_size]

    # DataLoaderها batch می‌سازند.
    # فقط train shuffle=True دارد تا ترتیب نمونه‌ها در هر epoch تغییر کند.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Data loaded!")
    print(f"Dataset split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    print(f"CLI target={args.target}, resolved target={actual_target} ({TARGET_NAMES.get(actual_target, 'unknown')})")
    print(f"Variant={args.variant}, weight_sharing={use_weight_sharing}, basis_reduction={use_basis_reduction}")

    # Config همه‌ی تنظیمات معماری را به models.py منتقل می‌کند.
    config = Config(
        dataset=args.dataset,
        dim=args.dim,
        n_layer=args.n_layer,
        cutoff_l=args.cutoff_l,
        cutoff_g=args.cutoff_g,
        use_weight_sharing=use_weight_sharing,
        use_basis_reduction=use_basis_reduction,
        num_spherical=args.num_spherical,
        num_radial=args.num_radial,
        num_rbf=args.num_rbf,
        reduced_num_spherical=args.reduced_num_spherical,
        reduced_num_radial=args.reduced_num_radial,
        reduced_num_rbf=args.reduced_num_rbf,
    )

    # انتخاب بین PAMNet کامل و PAMNet_s ساده‌تر.
    model = PAMNet(config).to(device) if args.model == "PAMNet" else PAMNet_s(config).to(device)
    num_params = count_parameters(model)
    print("Number of model parameters:", num_params)

    # Adam optimizer برای به‌روزرسانی وزن‌های مدل.
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)

    # scheduler اصلی: learning rate را به صورت exponential کم می‌کند.
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9961697)

    # warmup در ابتدای آموزش به کاهش ناپایداری شروع training کمک می‌کند.
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler)

    # EMA میانگین متحرک نمایی وزن‌ها را نگه می‌دارد و معمولاً evaluation را پایدارتر می‌کند.
    ema = EMA(model, decay=0.999)

    # ساخت نام run. اگر کاربر --run_name ندهد، نام شامل مدل، variant، target، seed و timestamp می‌شود.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.strip() or f"{args.model}_{args.variant}_target{args.target}_actual{actual_target}_seed{args.seed}_{timestamp}"

    # مسیر فایل‌های خروجی این run.
    os.makedirs(args.results_dir, exist_ok=True)
    epoch_csv_path = osp.join(args.results_dir, f"{run_name}_epochs.csv")
    summary_csv_path = osp.join(args.results_dir, "summary.csv")
    config_json_path = osp.join(args.results_dir, f"{run_name}_config.json")

    # ذخیره‌ی تنظیمات اجرا برای اینکه آزمایش بعداً قابل بازتولید باشد.
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print("Start training!")

    # زمان شروع کل آموزش.
    train_start_time = time.time()

    # بهترین validation MAE تا این لحظه. مقدار اولیه inf است تا epoch اول حتماً بهترین شود.
    best_val_mae = float("inf")
    best_test_mae = None
    best_epoch = None

    # حلقه‌ی اصلی آموزش.
    for epoch in range(args.epochs):
        # زمان شروع همین epoch برای گزارش زمان هر epoch.
        epoch_start_time = time.time()

        # loss_all جمع loss وزن‌دارشده با تعداد graphهای batch است.
        loss_all = 0.0
        step = 0

        # فعال کردن حالت train؛ روی لایه‌هایی مثل dropout/batchnorm اثر دارد.
        model.train()

        # اگر disable_tqdm فعال نباشد، progress bar نمایش داده می‌شود.
        iterator = train_loader if args.disable_tqdm else tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)

        for data in iterator:
            # انتقال batch به device.
            data = data.to(device)

            # پاک کردن gradientهای قبلی.
            optimizer.zero_grad()

            # forward pass: پیش‌بینی مقدار target برای moleculeهای batch.
            output = model(data)

            # L1 loss همان MAE batch است و برای QM9 معمولاً معیار اصلی گزارش است.
            loss = F.l1_loss(output, data.y)

            # جمع کردن loss برای محاسبه‌ی train MAE کل epoch.
            loss_all += loss.item() * data.num_graphs

            # backward pass: محاسبه‌ی gradientها.
            loss.backward()

            # gradient clipping جلوی انفجار gradient را می‌گیرد.
            clip_grad_norm_(model.parameters(), max_norm=1000, norm_type=2)

            # یک step به‌روزرسانی optimizer.
            optimizer.step()

            # curr_epoch مقدار پیوسته‌ی epoch است؛ برای warmup scheduler در سطح batch استفاده می‌شود.
            curr_epoch = epoch + float(step) / (len(train_dataset) / args.batch_size)
            scheduler_warmup.step(curr_epoch)

            # به‌روزرسانی EMA بعد از هر optimizer step.
            ema(model)
            step += 1

        # میانگین خطای train در این epoch.
        train_mae = loss_all / len(train_loader.dataset)

        # ارزیابی validation با وزن‌های EMA.
        val_mae = evaluate(model, val_loader, ema, device)

        # فقط وقتی validation بهتر شد test را حساب می‌کنیم؛ چون test نباید مبنای انتخاب مدل باشد.
        test_mae_if_best = ""
        if val_mae <= best_val_mae:
            best_val_mae = val_mae
            best_test_mae = evaluate(model, test_loader, ema, device)
            best_epoch = epoch + 1
            test_mae_if_best = best_test_mae

            # ذخیره‌ی بهترین checkpoint براساس validation.
            save_folder = osp.join(".", "save", args.dataset, run_name)
            os.makedirs(save_folder, exist_ok=True)
            torch.save(model.state_dict(), osp.join(save_folder, "best_model.pt"))

        # زمان این epoch.
        epoch_time = time.time() - epoch_start_time

        # ذخیره‌ی اطلاعات epoch در CSV برای رسم نمودار یا تحلیل بعدی.
        epoch_row = {
            "run_name": run_name,
            "epoch": epoch + 1,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "test_mae_if_best": test_mae_if_best,
            "best_val_mae_so_far": best_val_mae,
            "best_test_mae_so_far": best_test_mae,
            "best_epoch_so_far": best_epoch,
            "epoch_time_sec": epoch_time,
        }
        write_csv_row(epoch_csv_path, epoch_row)

        # چاپ گزارش epoch در terminal.
        print(
            "Epoch: {:03d}, Train MAE: {:.7f}, Val MAE: {:.7f}, Best Val MAE: {:.7f}, Best Test MAE: {:.7f}".format(
                epoch + 1,
                train_mae,
                val_mae,
                best_val_mae,
                best_test_mae if best_test_mae is not None else float("nan"),
            )
        )

    # زمان کل آموزش.
    total_time = time.time() - train_start_time

    # زمان inference نهایی روی test loader.
    inference_mae, inference_time = evaluate_with_time(model, test_loader, ema, device)

    # ثبت peak memory اگر CUDA فعال باشد.
    if device.type == "cuda":
        peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    else:
        peak_allocated_mb = 0.0
        peak_reserved_mb = 0.0

    # میانگین زمان هر epoch.
    avg_epoch_time = total_time / args.epochs

    # خلاصه‌ی نهایی run؛ این ردیف برای مقایسه‌ی چند variant کاربرد دارد.
    summary_row = {
        "run_name": run_name,
        "timestamp": timestamp,
        "dataset": args.dataset,
        "model": args.model,
        "variant": args.variant,
        "use_weight_sharing": use_weight_sharing,
        "use_basis_reduction": use_basis_reduction,
        "cli_target": args.target,
        "actual_target": actual_target,
        "target_name": TARGET_NAMES.get(actual_target, "unknown"),
        "atom_mapping_enabled": use_atom_mapping,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dim": args.dim,
        "n_layer": args.n_layer,
        "lr": args.lr,
        "weight_decay": args.wd,
        "cutoff_l": args.cutoff_l,
        "cutoff_g": args.cutoff_g,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "num_spherical_used": config.num_spherical,
        "num_radial_used": config.num_radial,
        "num_rbf_used": config.num_rbf,
        "num_parameters": num_params,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "best_test_mae": best_test_mae,
        "final_inference_mae": inference_mae,
        "total_train_time_sec": total_time,
        "total_train_time_min": total_time / 60.0,
        "avg_epoch_time_sec": avg_epoch_time,
        "test_inference_time_sec": inference_time,
        "peak_cuda_memory_allocated_mb": peak_allocated_mb,
        "peak_cuda_memory_reserved_mb": peak_reserved_mb,
        "device": str(device),
    }
    write_csv_row(summary_csv_path, summary_row)

    # چاپ خلاصه‌ی نهایی برای گزارش سریع در terminal.
    print("Best Validation MAE:", best_val_mae)
    print("Testing MAE at Best Validation:", best_test_mae)
    print(f"Total training time: {total_time:.2f} s ({total_time/60:.2f} min)")
    print(f"Average epoch time: {avg_epoch_time:.2f} s")
    print(f"Test inference time: {inference_time:.4f} s")
    print(f"Peak CUDA memory allocated: {peak_allocated_mb:.2f} MB")
    print(f"Peak CUDA memory reserved: {peak_reserved_mb:.2f} MB")
    print(f"Epoch log CSV: {epoch_csv_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Config JSON: {config_json_path}")


# اجرای main فقط وقتی فایل مستقیم اجرا شود؛ اگر این فایل import شود، main خودکار اجرا نمی‌شود.
if __name__ == "__main__":
    main()
