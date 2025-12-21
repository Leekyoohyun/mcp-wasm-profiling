#!/bin/bash
#
# setup_test_data.sh - Create test data for WASM MCP tool profiling
#
# This script generates various test files and directories for
# comprehensive tool performance measurement.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TEST_DATA_DIR="${PROJECT_DIR}/test_data"

echo "=== WASM MCP Profiling: Test Data Setup ==="
echo "Project directory: ${PROJECT_DIR}"
echo "Test data directory: ${TEST_DATA_DIR}"
echo ""

# =============================================================================
# Helper Functions
# =============================================================================

create_random_text() {
    local size=$1
    local output=$2

    echo "Creating ${size} byte text file: ${output}"

    # Use /dev/urandom for random data, but make it text-like
    # Each line is ~80 chars to simulate real text files
    local lines=$((size / 81))
    if [ $lines -lt 1 ]; then
        lines=1
    fi

    # Generate random alphanumeric text
    head -c $size /dev/urandom | base64 | head -c $size > "$output"
}

create_random_binary() {
    local size=$1
    local output=$2

    echo "Creating ${size} byte binary file: ${output}"
    head -c $size /dev/urandom > "$output"
}

create_directory_with_files() {
    local dir=$1
    local num_files=$2
    local file_size=${3:-1024}  # Default 1KB per file

    echo "Creating directory with ${num_files} files: ${dir}"
    mkdir -p "$dir"

    for i in $(seq 1 $num_files); do
        head -c $file_size /dev/urandom | base64 | head -c $file_size > "${dir}/file_$(printf '%04d' $i).txt"
    done
}

create_nested_directory() {
    local base_dir=$1
    local depth=$2
    local files_per_dir=${3:-5}

    echo "Creating nested directory (depth=${depth}): ${base_dir}"
    mkdir -p "$base_dir"

    local current_dir="$base_dir"
    for d in $(seq 1 $depth); do
        # Create files at this level
        for f in $(seq 1 $files_per_dir); do
            echo "File at depth $d, number $f" > "${current_dir}/file_${f}.txt"
        done
        # Create subdirectory
        current_dir="${current_dir}/level_${d}"
        mkdir -p "$current_dir"
    done

    # Create files in the deepest directory
    for f in $(seq 1 $files_per_dir); do
        echo "File at deepest level, number $f" > "${current_dir}/file_${f}.txt"
    done
}

# =============================================================================
# Create Directory Structure
# =============================================================================

echo "Creating directory structure..."
mkdir -p "${TEST_DATA_DIR}/files"
mkdir -p "${TEST_DATA_DIR}/images"
mkdir -p "${TEST_DATA_DIR}/directories"

# =============================================================================
# Create Test Files (Various Sizes)
# =============================================================================

echo ""
echo "=== Creating Test Files ==="

# Small files (KB range)
create_random_text 1024 "${TEST_DATA_DIR}/files/test_1KB.txt"
create_random_text 10240 "${TEST_DATA_DIR}/files/test_10KB.txt"
create_random_text 102400 "${TEST_DATA_DIR}/files/test_100KB.txt"

# Medium files (MB range)
create_random_text 1048576 "${TEST_DATA_DIR}/files/test_1MB.txt"
create_random_text 10485760 "${TEST_DATA_DIR}/files/test_10MB.txt"

# Large files (50MB+)
echo "Creating large test file (50MB)..."
create_random_text 52428800 "${TEST_DATA_DIR}/files/test_50MB.txt"

# Multiple file sets for read_multiple_files
echo ""
echo "Creating multiple file sets..."
for i in 0 1 2; do
    create_random_text 1024 "${TEST_DATA_DIR}/files/test_1KB_${i}.txt"
    create_random_text 102400 "${TEST_DATA_DIR}/files/test_100KB_${i}.txt"
    create_random_text 1048576 "${TEST_DATA_DIR}/files/test_1MB_${i}.txt"
done

# =============================================================================
# Create Test Images/Binary Files
# =============================================================================

echo ""
echo "=== Creating Test Binary Files (simulating images) ==="

create_random_binary 102400 "${TEST_DATA_DIR}/images/test_100KB.bin"
create_random_binary 1048576 "${TEST_DATA_DIR}/images/test_1MB.bin"
create_random_binary 5242880 "${TEST_DATA_DIR}/images/test_5MB.bin"
create_random_binary 10485760 "${TEST_DATA_DIR}/images/test_10MB.bin"

echo ""
echo "=== Creating Real PNG Images (for image-resize tools) ==="

# 2000x2000 PNG 이미지 생성 (PIL 사용) - 약 12MB
python3 -c "
from PIL import Image
import numpy as np

# 2000x2000 랜덤 컬러 이미지
pixels = np.random.randint(0, 256, (2000, 2000, 3), dtype=np.uint8)
img = Image.fromarray(pixels, 'RGB')
img.save('${TEST_DATA_DIR}/images/test.png')
print('Created 2000x2000 PNG: test.png (~12MB)')
" 2>/dev/null || echo "Warning: PIL/numpy not available, skipping PNG creation"

# =============================================================================
# Create Test Directories
# =============================================================================

echo ""
echo "=== Creating Test Directories ==="

# Small directory (10 files)
create_directory_with_files "${TEST_DATA_DIR}/directories/10files" 10

# Medium directory (100 files)
create_directory_with_files "${TEST_DATA_DIR}/directories/100files" 100

# Large directory (1000 files)
create_directory_with_files "${TEST_DATA_DIR}/directories/1000files" 1000 256

# Nested directories for directory_tree tests
echo ""
echo "Creating nested directory structures..."

# Shallow: depth 2
create_nested_directory "${TEST_DATA_DIR}/directories/shallow" 2 5

# Medium: depth 5
create_nested_directory "${TEST_DATA_DIR}/directories/medium" 5 5

# Deep: depth 10
create_nested_directory "${TEST_DATA_DIR}/directories/deep" 10 3

# Search test directories
echo ""
echo "Creating search test directories..."

# Small search directory
mkdir -p "${TEST_DATA_DIR}/directories/small_dir"
for i in $(seq 1 20); do
    ext=""
    case $((i % 4)) in
        0) ext="txt" ;;
        1) ext="json" ;;
        2) ext="md" ;;
        3) ext="log" ;;
    esac
    echo "Content of file $i" > "${TEST_DATA_DIR}/directories/small_dir/file_${i}.${ext}"
done

# Medium search directory
mkdir -p "${TEST_DATA_DIR}/directories/medium_dir"
for i in $(seq 1 200); do
    ext=""
    case $((i % 5)) in
        0) ext="txt" ;;
        1) ext="json" ;;
        2) ext="md" ;;
        3) ext="log" ;;
        4) ext="csv" ;;
    esac
    echo "Content of file $i" > "${TEST_DATA_DIR}/directories/medium_dir/file_${i}.${ext}"
done

# Large search directory with subdirectories
mkdir -p "${TEST_DATA_DIR}/directories/large_dir"
for subdir in src docs data config; do
    mkdir -p "${TEST_DATA_DIR}/directories/large_dir/${subdir}"
    for i in $(seq 1 50); do
        ext=""
        case $((i % 4)) in
            0) ext="txt" ;;
            1) ext="json" ;;
            2) ext="md" ;;
            3) ext="yaml" ;;
        esac
        echo "Content of file $i in $subdir" > "${TEST_DATA_DIR}/directories/large_dir/${subdir}/file_${i}.${ext}"
    done
done

# =============================================================================
# Create Test Git Repository
# =============================================================================

echo ""
echo "=== Creating Test Git Repository (Large) ==="

GIT_REPO_DIR="${TEST_DATA_DIR}/git_repo"

# 기존 git repo 삭제 후 새로 생성
rm -rf "${GIT_REPO_DIR}"
mkdir -p "${GIT_REPO_DIR}"
cd "${GIT_REPO_DIR}"

# git 초기화
git init

# 테스트용 user 설정 (repo 로컬)
git config user.email "test@test.com"
git config user.name "Test User"

# 초기 파일 생성 - 100KB README
head -c 102400 /dev/urandom | base64 > README.md
git add README.md
git commit -m "Initial commit with large README"

# 50개 파일 생성 (각 10KB)
echo "Creating 50 source files..."
mkdir -p src
for i in $(seq 1 50); do
    head -c 10240 /dev/urandom | base64 > "src/module_${i}.py"
done
git add src/
git commit -m "Add 50 source modules"

# 50개 추가 커밋 (파일 수정)
echo "Creating 50 commits with modifications..."
for i in $(seq 1 50); do
    # 랜덤 파일 수정
    file_num=$((($i % 50) + 1))
    echo "# Modified in commit $i" >> "src/module_${file_num}.py"
    head -c 1024 /dev/urandom | base64 >> "src/module_${file_num}.py"
    git add "src/module_${file_num}.py"
    git commit -m "Commit $i: modify module_${file_num}"
done

# 큰 바이너리 파일 추가 (1MB)
echo "Adding large binary file..."
mkdir -p assets
head -c 1048576 /dev/urandom > assets/large_binary.bin
git add assets/
git commit -m "Add large binary asset"

# Pack 파일 생성 (loose objects를 pack으로 변환)
echo "Running git gc to create pack files..."
git gc --aggressive

# 브랜치 생성
git branch feature-branch
git branch bugfix-branch
git branch develop

# unstaged 변경사항 (git_diff_unstaged 테스트용)
echo "# Unstaged modification" >> src/module_1.py
head -c 5120 /dev/urandom | base64 >> src/module_1.py

# staged 변경사항 (git_diff_staged 테스트용)
echo "staged content with more data" > staged.txt
head -c 10240 /dev/urandom | base64 >> staged.txt
git add staged.txt

# untracked 파일 (git_status 테스트용)
echo "untracked file content" > untracked.txt

cd "${PROJECT_DIR}"

echo "Git repository created at: ${GIT_REPO_DIR}"
echo "  - 52+ commits"
echo "  - 50+ source files (10KB each)"
echo "  - 1 large binary file (1MB)"
echo "  - Pack files created (git gc)"
echo "  - 3 branches"
echo "  - 1 unstaged change"
echo "  - 1 staged change"
echo "  - 1 untracked file"

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=== Test Data Setup Complete ==="
echo ""
echo "Files created:"
echo "  Text files: 1KB, 10KB, 100KB, 1MB, 10MB, 50MB"
echo "  Binary files: 100KB, 1MB, 5MB, 10MB"
echo "  Multi-file sets: 3x1KB, 3x100KB, 3x1MB"
echo ""
echo "Directories created:"
echo "  Flat: 10files, 100files, 1000files"
echo "  Nested: shallow (depth 2), medium (depth 5), deep (depth 10)"
echo "  Search: small_dir (20 files), medium_dir (200 files), large_dir (4 subdirs)"
echo ""

# Calculate total size
echo "Total test data size:"
du -sh "${TEST_DATA_DIR}"

echo ""
echo "You can now run: python3 scripts/measure_tools.py"
