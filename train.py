from __future__ import annotations
import argparse, json, os, random, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
for key, folder in {"MPLCONFIGDIR": "matplotlib", "HF_HOME": "huggingface", "TORCH_HOME": "torch"}.items():
    os.environ.setdefault(key, str(ROOT / ".cache" / folder))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

ImageFile.LOAD_TRUNCATED_IMAGES = True
DATA_ROOT, ARTIFACTS, SEED = ROOT / "data" / "Confirmed_fronts" / "confirmed_fronts", ROOT / "artifacts", 42


#Фиксация генераторов случайных чисел
def seed_everything(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


#Построение manifest: разбор имён DVM, нормализация цветов и group-split
def make_manifest(force: bool = False) -> pd.DataFrame:
    path = ARTIFACTS / "manifest.csv"
    if path.exists() and not force: return pd.read_csv(path)

    #Из имени каждого JPEG извлекаются путь, марка, модель, год, цвет и идентификатор объявления
    rows = []
    for p in DATA_ROOT.rglob("*.jpg"):
        x = p.stem.split("$$")
        if len(x) >= 7:
            rows.append(dict(path=str(p.resolve()), brand=x[0], model=x[1], year=x[2], color=x[3].strip().title(), group=f"{x[4]}$${x[5]}"))
    df = pd.DataFrame(rows).drop_duplicates("path").reset_index(drop=True)
    if df.empty: raise FileNotFoundError(f"No DVM JPG files found under {DATA_ROOT}")

    #Служебный Unlisted удаляется, а синонимы объединяются в визуальные классы
    aliases = {"Burgundy": "Red", "Maroon": "Red", "Navy": "Blue", "Indigo": "Blue", "Magenta": "Purple", "Turquoise": "Blue"}
    df["color_raw"] = df.color
    df = df[df.color.ne("Unlisted")].copy().reset_index(drop=True)
    df["color"] = df.color.replace(aliases)

    #Неоднозначные группы уточняются цветом, затем классам назначаются целочисленные индексы
    if (df.groupby("group").color.nunique() > 1).any(): df["group"] += "$$" + df.color
    classes = sorted(df.color.unique()); df["target"] = df.color.map({c: i for i, c in enumerate(classes)})

    #StratifiedGroupKFold сохраняет баланс классов и не разделяет одно объявление между выборками
    cv, df["fold"] = StratifiedGroupKFold(10, shuffle=True, random_state=SEED), -1
    for fold, (_, idx) in enumerate(cv.split(df, df.target, groups=df.group)): df.loc[idx, "fold"] = fold
    df["split"] = np.where(df.fold.eq(0), "test", np.where(df.fold.eq(1), "val", "train"))

    #Manifest и порядок классов сохраняются для всех трёх одинаковых экспериментов
    ARTIFACTS.mkdir(exist_ok=True); df.to_csv(path, index=False)
    (ARTIFACTS / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    return df


#Dataset открывает RGB-изображение и возвращает тензор, номер класса и путь
class CarDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame, self.transform = frame.reset_index(drop=True), transform

    #Размер датасета равен числу строк соответствующего split в manifest
    def __len__(self): return len(self.frame)

    #Один объект загружается с диска, преобразуется и связывается с целевой меткой
    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        with Image.open(row.path) as im: image = im.convert("RGB")
        return self.transform(image), int(row.target), row.path


#Train получает безопасные для цвета аугментации, а validation/test только resize и нормализацию
def make_transforms(size: int, mean, std):
    common = [transforms.Resize((size, size), antialias=True), transforms.ToTensor(), transforms.Normalize(mean, std)]
    train_tf = transforms.Compose([ common[0], transforms.RandomHorizontalFlip(), transforms.RandomApply([transforms.RandomAffine(4, translate=(.025, .025), scale=(.94, 1.04))], p=.5), transforms.ColorJitter(brightness=.12, contrast=.10, saturation=.05, hue=.01), *common[1:], transforms.RandomErasing(p=.08, scale=(.02, .08), ratio=(.5, 2.), value=0)])
    return train_tf, transforms.Compose(common)


#DataLoader создаются для train/val/test; test и validation используют удвоенный batch без перемешивания
def make_loaders(df, size, mean, std, batch_size, workers=4):
    train_tf, eval_tf = make_transforms(size, mean, std)
    kw = dict(num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)
    return {"train": DataLoader(CarDataset(df[df.split == "train"], train_tf), batch_size=batch_size, shuffle=True, drop_last=True, **kw), **{s: DataLoader(CarDataset(df[df.split == s], eval_tf), batch_size=batch_size * 2, shuffle=False, **kw) for s in ("val", "test")}}


#SE-блок оценивает важность каналов и усиливает полезные цветовые и текстурные признаки
class SE(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__(); hidden = max(8, channels // reduction)
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    #Карта признаков умножается на обученные поканальные коэффициенты внимания
    def forward(self, x): return x * self.net(x)


#ResidualSEBlock объединяет две свёртки, SE-внимание и skip-связь для устойчивого обучения
class ResidualSEBlock(nn.Module):
    def __init__(self, inp: int, out: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(inp, out, 3, stride, 1, bias=False), nn.BatchNorm2d(out), nn.SiLU(), nn.Conv2d(out, out, 3, 1, 1, bias=False), nn.BatchNorm2d(out), SE(out))
        self.skip = (nn.Identity() if inp == out and stride == 1 else nn.Sequential(nn.Conv2d(inp, out, 1, stride, bias=False), nn.BatchNorm2d(out)))
        self.act = nn.SiLU()

    #Результат свёрточной ветки складывается с исходным или проецированным входом
    def forward(self, x): return self.act(self.body(x) + self.skip(x))


#Собственная ColorAwareCNN совмещает SE-ResNet признаки и явные глобальные RGB-статистики
class ColorAwareCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__(); widths = [32, 48, 80, 128, 192]
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 5, 2, 2, bias=False), nn.BatchNorm2d(widths[0]), nn.SiLU())
        blocks = [block for a, b in zip(widths, widths[1:]) for block in (ResidualSEBlock(a, b, 2), ResidualSEBlock(b, b))]
        self.features, self.pool = nn.Sequential(*blocks), nn.AdaptiveAvgPool2d(1)
        self.color_mlp = nn.Sequential(nn.Linear(6, 32), nn.BatchNorm1d(32), nn.SiLU(), nn.Dropout(.1))
        self.classifier = nn.Sequential(nn.Dropout(.3), nn.Linear(widths[-1] + 32, num_classes))

    #CNN-вектор объединяется со средним и стандартным отклонением RGB, затем классифицируется
    def forward(self, x):
        visual = self.pool(self.features(self.stem(x))).flatten(1)
        color = self.color_mlp(torch.cat([x.mean((2, 3)), x.std((2, 3))], 1))
        return self.classifier(torch.cat([visual, color], 1))


#Обёртка CLIP добавляет к визуальному энкодеру нормализацию, dropout и новый цветовой head
class ClipClassifier(nn.Module):
    def __init__(self, clip_model, feature_dim: int, num_classes: int):
        super().__init__(); self.clip = clip_model
        self.head = nn.Sequential(nn.LayerNorm(feature_dim), nn.Dropout(.15), nn.Linear(feature_dim, num_classes))

    #CLIP формирует визуальный вектор, который новый head переводит в логиты цветов
    def forward(self, x): return self.head(self.clip.encode_image(x, normalize=False))


#Фабрика создаёт выбранную модель и возвращает неизменные preprocessing- и train-гиперпараметры
def build_model(name: str, num_classes: int):
    imagenet_norm = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    #Собственная модель обучается со случайной инициализации 10 эпох на изображениях 160×160
    if name == "scratch": return ColorAwareCNN(num_classes), 160, *imagenet_norm, 72, 10, 8e-4

    #EfficientNet получает ImageNet-веса; заморожены все блоки кроме последних трёх и нового head
    if name == "efficientnet_imagenet":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        for p in model.features.parameters(): p.requires_grad = False
        for block in list(model.features.children())[-3:]:
            for p in block.parameters(): p.requires_grad = True
        return model, 192, *imagenet_norm, 80, 5, 4e-4

    #CLIP получает LAION-веса; обучаются последний visual-блок, ln_post, projection и новый head
    if name == "clip_laion":
        import open_clip
        clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="laion400m_e32")
        for p in clip.parameters(): p.requires_grad = False
        for module in (clip.visual.transformer.resblocks[-1], clip.visual.ln_post):
            for p in module.parameters(): p.requires_grad = True
        if isinstance(clip.visual.proj, nn.Parameter): clip.visual.proj.requires_grad = True
        model = ClipClassifier(clip, clip.visual.output_dim, num_classes)
        return model, 224, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711), 64, 3, 3e-5
    raise ValueError(name)


#Инференс без градиентов собирает истинные метки, предсказания и пути ко всем изображениям
@torch.inference_mode()
def predict(model, loader, device):
    model.eval(); ys, ps, paths = [], [], []
    for x, y, p in loader:
        logits = model(x.to(device, non_blocking=True)); ys += y.tolist(); ps += logits.argmax(1).cpu().tolist(); paths += list(p)
    return np.asarray(ys), np.asarray(ps), paths


#Одна train-эпоха выполняет forward, weighted loss, backward, clipping градиентов и шаг оптимизатора
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp=True):
    model.train(); total_loss = n = 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp and device.type == "cuda"):
            logits = model(x); loss = criterion(logits, y)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), 5.)
        scaler.step(optimizer); scaler.update(); total_loss += loss.item() * len(y); n += len(y)
    return total_loss / n


#Итоговая диагностика сохраняет accuracy, macro-F1, per-class отчёт, предсказания и confusion matrix
def save_diagnostics(name, classes, y_true, y_pred, paths, history):
    out = ARTIFACTS / name; out.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_true, y_pred, labels=range(len(classes)), target_names=classes,
                                   output_dict=True, zero_division=0)
    metrics = dict(model=name, accuracy=float(accuracy_score(y_true, y_pred)),
                   f1_macro=float(f1_score(y_true, y_pred, average="macro")), n_test=int(len(y_true)),
                   history=history, classification_report=report)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(dict(path=paths, y_true=y_true, y_pred=y_pred)).to_csv(out / "predictions.csv", index=False)
    np.save(out / "confusion_matrix.npy", confusion_matrix(y_true, y_pred, labels=range(len(classes)), normalize="true"))
    return metrics


#Главный эксперимент связывает данные, модель, weighted loss, validation-отбор и финальный test
def run(name: str):
    seed_everything(); ARTIFACTS.mkdir(exist_ok=True); df = make_manifest()
    classes = json.loads((ARTIFACTS / "classes.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, size, mean, std, batch_size, epochs, lr = build_model(name, len(classes))
    loaders = make_loaders(df, size, mean, std, batch_size); model.to(device)

    #Обучаются только requires_grad-параметры; веса классов равны корню из обратной частоты
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 30)
    counts = df[df.split == "train"].target.value_counts().sort_index().values
    weights = torch.tensor(np.sqrt(counts.sum() / (len(counts) * counts)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=.04)
    use_amp = False
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    #На каждой эпохе считается validation macro-F1 и сохраняется лучший checkpoint
    out = ARTIFACTS / name; out.mkdir(parents=True, exist_ok=True); best_f1, history = -1., []
    print(f"{name}: device={device}, trainable={sum(p.numel() for p in trainable):,}, samples={len(df):,}", flush=True)
    for epoch in range(1, epochs + 1):
        start = time.time(); loss = train_one_epoch(model, loaders["train"], optimizer, criterion, scaler, device, use_amp)
        y_val, p_val, _ = predict(model, loaders["val"], device); val_f1 = f1_score(y_val, p_val, average="macro")
        row = dict(epoch=epoch, train_loss=loss, val_f1_macro=float(val_f1), seconds=time.time() - start)
        history.append(row); print(json.dumps(row), flush=True)
        if val_f1 > best_f1:
            best_f1 = val_f1; torch.save(dict(state_dict=model.state_dict(), classes=classes, model=name), out / "best.pt")
        scheduler.step()

    #Лучший checkpoint оценивается один раз на test, после чего печатаются главные метрики
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True)["state_dict"])
    y_test, p_test, paths = predict(model, loaders["test"], device)
    metrics = save_diagnostics(name, classes, y_test, p_test, paths, history)
    print(json.dumps({k: metrics[k] for k in ("model", "accuracy", "f1_macro", "n_test")}, indent=2))


#CLI выбирает одну из трёх моделей и при необходимости перестраивает manifest перед обучением
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("scratch", "efficientnet_imagenet", "clip_laion"), required=True)
    parser.add_argument("--rebuild-manifest", action="store_true"); args = parser.parse_args()
    if args.rebuild_manifest: ARTIFACTS.mkdir(exist_ok=True); make_manifest(force=True)
    run(args.model)
