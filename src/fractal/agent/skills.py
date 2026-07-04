from __future__ import annotations

from predict_rlm import Skill

FILESYSTEM_CODING_INSTRUCTIONS = """
# Python filesystem cheatsheet for an agent

Use `os` for syscall-like filesystem operations: directory-relative paths, raw file descriptors, offset reads/writes, truncation, renaming, deleting, and directory listing.

```python
import os
import stat

# Open the trusted workspace directory from the `workspace` input variable.
# Treat this as the root for all relative paths.
root_fd = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY)
```

## Searching text

Use `rg` through `subprocess.run` for text search. Do not implement grep by
recursively reading every file in Python.

Useful `rg` flags:

```text
rg -n PATTERN PATH...       # line-numbered matches: path:line:text
rg -l PATTERN PATH...       # matching file paths only
rg -c PATTERN PATH...       # match counts per file
rg -C 3 PATTERN PATH...     # include 3 context lines around matches
rg --json PATTERN PATH...   # machine-readable JSON events
```

```python
import subprocess

result = subprocess.run(
    ["rg", "-n", "RuntimeHook|runtime_hook", "src", "tests"],
    cwd=workspace,
    text=True,
    capture_output=True,
)

matches = []
for line in result.stdout.splitlines():
    path, line_number, text = line.split(":", 2)
    matches.append((path, int(line_number), text.strip()))
```

Prefer searching focused source directories such as `src`, `tests`, `docs`, or
specific included project roots. Avoid broad recursive Python walks like
`Path(root).rglob("*.py")` unless you must inspect filesystem metadata. If you
do walk manually, skip dependency, cache, VCS, sandbox, and session directories:
`.git`, `.venv`, `node_modules`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`,
`.predict_rlm_sbx`, `.predict_rlm_runner_env`, and `.fractal`.

## Opening & creating files

```python
# Open existing file read-only.
fd = os.open("file.txt", os.O_RDONLY, dir_fd=root_fd)

# Open existing file for reading and writing.
fd = os.open("file.txt", os.O_RDWR, dir_fd=root_fd)

# Open existing file for writing only.
fd = os.open("file.txt", os.O_WRONLY, dir_fd=root_fd)

# Create a new file; fail if it already exists.
fd = os.open(
    "new_file.txt",
    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
    0o644,
    dir_fd=root_fd,
)

# Create file if needed, or truncate existing file to zero bytes.
fd = os.open(
    "output.txt",
    os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
    0o644,
    dir_fd=root_fd,
)

# Open a directory.
dir_fd = os.open("src", os.O_RDONLY | os.O_DIRECTORY, dir_fd=root_fd)
```

## Reading files

```python
# Read up to 4096 bytes from offset 0.
data = os.pread(fd, 4096, 0)

# Read 1024 bytes starting at byte offset 5000.
chunk = os.pread(fd, 1024, 5000)

# Read a whole file after opening it.
size = os.fstat(fd).st_size
data = os.pread(fd, size, 0)
```

## Writing bytes

```python
# Write bytes at the beginning of a file.
os.pwrite(fd, b"hello", 0)

# Overwrite bytes in the middle of a file.
# This replaces existing bytes; it does not insert.
os.pwrite(fd, b"XYZ", 10)

# Append by writing at the current file size.
end = os.fstat(fd).st_size
os.pwrite(fd, b"\nnew line\n", end)

# Ensure file contents are flushed to disk.
os.fsync(fd)
```

## Writing a full buffer

```python
# pwrite may write fewer bytes than requested, so loop for larger writes.
data = b"complete file contents\n"

offset = 0
while offset < len(data):
    written = os.pwrite(fd, data[offset:], offset)
    if written <= 0:
        raise OSError("pwrite made no progress")
    offset += written

os.fsync(fd)
```

## Resizing files

```python
# Empty a file.
os.ftruncate(fd, 0)

# Resize a file to exactly 1024 bytes.
os.ftruncate(fd, 1024)

# Grow a file to 1 MiB.
os.ftruncate(fd, 1024 * 1024)

os.fsync(fd)
```

## Whole-file replacement

Use this when replacing a file with new content, especially when the new content has a different size.

```python
# Write new contents into a temporary file.
tmp = ".file.txt.tmp"
dst = "file.txt"

tmp_fd = os.open(
    tmp,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_TRUNC,
    0o644,
    dir_fd=root_fd,
)

data = b"new complete file contents\n"

offset = 0
while offset < len(data):
    written = os.pwrite(tmp_fd, data[offset:], offset)
    if written <= 0:
        raise OSError("pwrite made no progress")
    offset += written

os.fsync(tmp_fd)

# Atomically replace the destination with the temp file.
os.replace(
    tmp,
    dst,
    src_dir_fd=root_fd,
    dst_dir_fd=root_fd,
)

# Flush the directory entry so the rename is durable.
os.fsync(root_fd)
```

## Mid-file insert/delete/replace

There is no native "insert bytes here" syscall. Read the original bytes, build the new bytes, then use whole-file replacement.

```python
# Read original file.
fd = os.open("file.txt", os.O_RDONLY, dir_fd=root_fd)
size = os.fstat(fd).st_size
original = os.pread(fd, size, 0)

# Insert bytes at offset 10.
start = 10
updated = original[:start] + b"INSERTED" + original[start:]

# Delete 5 bytes at offset 10.
start = 10
old_len = 5
updated = original[:start] + original[start + old_len:]

# Replace 5 bytes at offset 10.
start = 10
old_len = 5
updated = original[:start] + b"replacement" + original[start + old_len:]

# Then write `updated` using the whole-file replacement pattern.
```

## Metadata

```python
# Stat a path relative to the workspace.
st = os.stat("file.txt", dir_fd=root_fd, follow_symlinks=False)

size = st.st_size
is_file = stat.S_ISREG(st.st_mode)
is_dir = stat.S_ISDIR(st.st_mode)

# Stat an already-open file descriptor.
st = os.fstat(fd)
size = st.st_size
```

## Directories

```python
# Create a directory.
os.mkdir("src", 0o755, dir_fd=root_fd)
os.fsync(root_fd)

# Open a directory.
dir_fd = os.open("src", os.O_RDONLY | os.O_DIRECTORY, dir_fd=root_fd)

# List workspace directory.
with os.scandir(root_fd) as entries:
    for entry in entries:
        print(entry.name)

# List a subdirectory.
with os.scandir(dir_fd) as entries:
    for entry in entries:
        print(entry.name, entry.is_file(), entry.is_dir())
```

## Rename & move

```python
# Rename a file.
os.replace(
    "old_name.txt",
    "new_name.txt",
    src_dir_fd=root_fd,
    dst_dir_fd=root_fd,
)

os.fsync(root_fd)

# Move a file into a subdirectory.
os.replace(
    "main.py",
    "src/main.py",
    src_dir_fd=root_fd,
    dst_dir_fd=root_fd,
)

os.fsync(root_fd)
```

## Delete

```python
# Delete a file.
os.unlink("old_file.txt", dir_fd=root_fd)
os.fsync(root_fd)

# Delete an empty directory.
os.rmdir("empty_dir", dir_fd=root_fd)
os.fsync(root_fd)
```

## Cleanup

```python
# Close file descriptors when finished with them.
os.close(fd)
os.close(dir_fd)
os.close(root_fd)
```

## Agent rules

- Use paths relative to `root_fd`.
- Do not use absolute paths.
- Do not use `..`.
- Use `os.pread` for offset reads.
- Use `os.pwrite` for offset overwrites.
- Use `os.ftruncate` for resizing.
- Use temp-file plus `os.replace` for safe whole-file replacement.
- For insert/delete/mid-file edits, read, build updated bytes, then replace the whole file.
"""


filesystem_coding_skill = Skill(
    name="filesystem-coding",
    instructions=FILESYSTEM_CODING_INSTRUCTIONS,
)
