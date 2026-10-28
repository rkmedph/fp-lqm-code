#!/bin/sh
# Переписывает абсолютные пути рабочей среды автора на пути этого каталога.
# Запустить один раз после распаковки, из корня пакета:   sh set_paths.sh
# Код намеренно сохранён в том виде, в котором он произвёл опубликованные
# результаты; меняются только два префикса пути.
set -e
ROOT=$(cd "$(dirname "$0")" && pwd)
find "$ROOT" -name '*.py' -type f -print0 \
  | xargs -0 sed -i "s#/home/claude/pmb#$ROOT/core#g; s#/home/claude/rawdata#$ROOT/data#g"
echo "Пути настроены на $ROOT"
echo "Осталось вхождений /home/claude: $(grep -rc '/home/claude' --include='*.py' "$ROOT" | awk -F: '{s+=$2} END {print s+0}')"
