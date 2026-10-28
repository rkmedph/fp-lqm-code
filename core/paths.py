"""Корни проекта и проверки входных данных для скриптов сборки и проверок.

Ни один скрипт не зашивает абсолютные пути. Корень определяется в таком порядке:
  1. аргумент командной строки --root PATH;
  2. переменная окружения FPLQM_ROOT;
  3. поиск вверх от файла скрипта до каталога, содержащего discrete.py.

Корень данных (каталог с подкаталогом analysis/) --- аналогично через --data,
FPLQM_DATA или поиск среди ../rawdata, ./data, ../data относительно корня проекта.

Использование в скрипте:

    import sys
    from pathlib import Path
    for _p in Path(__file__).resolve().parents:
        if (_p / "paths.py").exists():
            sys.path.insert(0, str(_p)); break
    from paths import ROOT, data_root, require, load_json
"""
import json
import os
import sys
from pathlib import Path

__all__ = ["ROOT", "DATA", "data_root", "require", "load_json", "argv_rest"]

_MARKER = "discrete.py"


def _cli(flag):
    """Достаёт --flag VALUE из sys.argv и убирает его оттуда."""
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 >= len(sys.argv):
            raise SystemExit(f"{flag} требует значения")
        val = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return val
    for i, a in enumerate(sys.argv):
        if a.startswith(flag + "="):
            del sys.argv[i]
            return a[len(flag) + 1:]
    return None


def _resolve_root():
    for src, val in (("--root", _cli("--root")), ("FPLQM_ROOT", os.environ.get("FPLQM_ROOT"))):
        if val:
            p = Path(val).expanduser().resolve()
            if not (p / _MARKER).exists():
                raise SystemExit(f"{src}={p}: в этом каталоге нет {_MARKER}")
            return p
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / _MARKER).exists():
            return p
    raise SystemExit(
        f"не найден корень проекта: ни в одном родительском каталоге {here} нет {_MARKER}.\n"
        f"Укажите его явно: --root PATH или FPLQM_ROOT=PATH")


ROOT = _resolve_root()


def _resolve_data():
    for src, val in (("--data", _cli("--data")), ("FPLQM_DATA", os.environ.get("FPLQM_DATA"))):
        if val:
            p = Path(val).expanduser().resolve()
            if not (p / "analysis").is_dir():
                raise SystemExit(f"{src}={p}: в этом каталоге нет подкаталога analysis/")
            return p
    for c in (ROOT.parent / "rawdata", ROOT / "data", ROOT.parent / "data"):
        if (c / "analysis").is_dir():
            return c.resolve()
    return None


DATA = _resolve_data()


def data_root():
    """Корень данных; завершает работу с внятным сообщением, если его нет."""
    if DATA is None:
        raise SystemExit(
            "не найден каталог данных с подкаталогом analysis/ "
            f"(искал {ROOT.parent/'rawdata'}, {ROOT/'data'}, {ROOT.parent/'data'}).\n"
            "Укажите его явно: --data PATH или FPLQM_DATA=PATH")
    return DATA


def require(*paths, what="вход"):
    """Проверяет существование всех путей сразу и сообщает обо всех отсутствующих."""
    paths = [Path(p) for p in paths]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"не найден {what}:\n" + "\n".join(f"  {p}" for p in missing))
    return paths[0] if len(paths) == 1 else paths


def load_json(path, *, require_keys=(), unique_by=None, nonempty=True):
    """Читает JSON и проверяет структуру до начала вычислений.

    require_keys --- ключи, обязательные в объекте верхнего уровня (для dict)
                     либо в каждом элементе (для list).
    unique_by    --- поле, по которому элементы списка должны быть уникальны.
    """
    path = Path(path)
    require(path, what="файл JSON")
    try:
        obj = json.loads(path.read_text(encoding="utf8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"{path}: некорректный JSON ({e})")
    if nonempty and not obj:
        raise SystemExit(f"{path}: пустой JSON")
    if isinstance(obj, dict):
        miss = [k for k in require_keys if k not in obj]
        if miss:
            raise SystemExit(f"{path}: нет обязательных ключей {miss}")
    elif isinstance(obj, list):
        for i, el in enumerate(obj):
            miss = [k for k in require_keys if k not in el]
            if miss:
                raise SystemExit(f"{path}: в элементе {i} нет ключей {miss}")
        if unique_by:
            seen, dup = set(), []
            for el in obj:
                v = el.get(unique_by)
                (dup.append(v) if v in seen else seen.add(v))
            if dup:
                raise SystemExit(f"{path}: повторяющиеся {unique_by}: {sorted(set(dup))}")
    return obj


def argv_rest():
    """Оставшиеся аргументы после снятия --root/--data."""
    return sys.argv[1:]
