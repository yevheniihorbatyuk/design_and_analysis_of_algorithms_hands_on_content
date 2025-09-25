#!/bin/bash

# Задаємо діапазон тижнів
START_WEEK=0
END_WEEK=6

# Створюємо основні файли
touch README.md
touch syllabus.md

# Створюємо папку resources та її вміст
mkdir -p resources
touch resources/articles.md
touch resources/books.md
touch resources/links.md

# Створюємо папку utils та її вміст
mkdir -p utils
touch utils/setup_env.md
touch utils/docker-compose.yml
touch utils/requirements.txt

# Створюємо папку projects та підпапки для кожного тижня
mkdir -p projects

# Цикл для створення папок тижнів
for (( week=START_WEEK; week<=END_WEEK; week++ ))
do
  WEEK_DIR="projects/week_$(printf "%02d" $week)"
  mkdir -p "$WEEK_DIR"
  touch "$WEEK_DIR/README.md"
  mkdir -p "$WEEK_DIR/data"
  mkdir -p "$WEEK_DIR/src"
done

echo "Структуру папок і файлів для курсу успішно створено!"
