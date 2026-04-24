---
title: "Linux Basics: Core Concepts and Essential Commands"
date: 2024-01-15 09:00:00
tags:
  - Linux
  - Cloud
  - System Administration
categories: Linux
series: "Linux"
series_order: 1
series_total: 8
lang: en
mathjax: false
description: "Your entry guide to Linux: the multi-user permission model, the FHS directory tree, distribution lineages, and the command-line muscle memory you need before any deeper topic makes sense."
disableNunjucks: true
---

The "difficulty" of Linux rarely lives in the commands themselves. The hard part is whether you have a clear *map* of the system: why it dominates servers, what multi-user and per-file permissions actually buy you, what changes when you switch between Debian and Red Hat lineages, and what to do in the first ten minutes after an SSH prompt opens. This post is the **entry guide** for the entire Linux series. It first builds the mental model -- philosophy, distributions, the FHS tree -- and then walks you through the commands you will use ten times an hour: `cd ls pwd`, `cp mv rm mkdir`, `cat less head tail`, `find grep`, plus pipelines, redirection, SSH, and a quick taste of permissions and processes. Each topic is intentionally **kept short**; depth lives in the dedicated articles (File Permissions, Disk Management, User Management, Service Management, Process Management, Package Management, Advanced File Operations).

# Why Linux, and Why It Looks the Way It Does

Three design decisions explain almost every Linux quirk a newcomer notices: it was built for **many users at once**, it treats **files as the universal interface**, and it expects **automation over clicking**.

- **Open and customisable.** Every component, from the kernel to the init system, can be swapped, recompiled, or stripped down. A 5 MB Alpine container and a 12 GB Oracle Linux install both call themselves Linux.
- **Stable enough to forget about.** Production servers routinely run for years without a reboot. The `uptime` command on a long-lived machine printing `up 412 days` is a normal sight, not a brag.
- **A package manager is the primary install path.** You almost never download a `.exe`. `apt`, `dnf`, `pacman`, `zypper` resolve dependencies, verify signatures, and let you upgrade the entire system with one command.
- **Everything is a file.** Disks live in `/dev`, processes appear in `/proc`, kernel knobs are toggled by writing to `/sys`. The same `cat` and `>` operators read CPU info or set LED brightness.
- **CLI first, GUI optional.** Graphical desktops exist (GNOME, KDE), but on servers you connect over SSH and drive the box with text -- which is also what makes scripting and remote management trivial.

## Distribution Families at a Glance

Most distros descend from a small number of ancestors. The package manager is usually the fastest way to tell them apart.

![Linux distribution family tree](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/basics/fig5_distro_family_tree.png)

- **Debian / Ubuntu** -- `apt`. The friendliest learning curve, vast documentation, and the default for most cloud images. Pick **Ubuntu LTS** if you have no other constraint.
- **Red Hat / RHEL / CentOS / Rocky / Alma / Fedora** -- `yum` (CentOS 7) or `dnf` (8+). The enterprise default. After CentOS Linux was discontinued in 2021, **Rocky Linux** and **AlmaLinux** became the binary-compatible drop-in replacements.
- **SUSE / openSUSE** -- `zypper`. Common in European enterprises and SAP shops.
- **Arch / Manjaro** -- `pacman`. Rolling release, latest packages, expects you to read the wiki.
- **Independents** -- Gentoo (compile everything), Alpine (musl + 5 MB, container darling), NixOS (declarative config), Void (no systemd).

For cloud, also check what your provider blesses: AWS ships **Amazon Linux**, Alibaba Cloud ships **Alibaba Cloud Linux**, both are RHEL derivatives tuned for the platform.

## The Three Ideas That Explain Everything Else

### 1. Multi-user, multi-task

Dozens of users may be logged in at once over SSH or local TTYs, each running many processes in parallel. The kernel must isolate their CPU time, memory, files, and network sockets. This is the reason the permission model is strict: without it, any user could read another user's secrets or kill another user's processes.

### 2. File-centric permissions

Every file (and a directory is just a special file) has three permission groups -- **owner**, **group**, **others** -- and three bits each: **read (r)**, **write (w)**, **execute (x)**. Read a permission string left to right:

```text
-rwxr-xr-x  1 alice  devs   2048  Jan 15 09:30  deploy.sh
^^^^^^^^^^
||||||||||+-- others:  r-x       can read and execute
|||||||+----- group:   r-x       can read and execute
||||+-------- owner:   rwx       can read, write, execute
+------------ type:    -         regular file (d=dir, l=symlink)
```

A common pattern you should recognise immediately: `rw-------` (mode `600`) is the only permission an SSH private key is allowed to have -- the SSH client refuses to use it otherwise.

> Depth on `chmod`/`chown`, numeric vs symbolic notation, SUID/SGID/sticky bit, ACLs, and `umask` lives in the **Linux File Permissions** article. The basics here are enough to read what `ls -l` shows you.

### 3. Everything is a file

Regular files, directories, devices, processes, kernel state, pipes, sockets -- all expose the same `read()` / `write()` interface. The payoff is that one set of tools works everywhere:

```bash
cat /proc/cpuinfo                        # CPU model, cores, flags
cat /proc/meminfo | head -5              # memory totals
echo 1 > /sys/class/leds/input1::capslock/brightness   # turn on capslock LED
cat /dev/urandom | head -c 16 | xxd      # 16 random bytes, hex-dumped
```

If you can `cat` it and `>` to it, you can usually script it.

# The Filesystem Map (FHS)

Unlike Windows, there are no drive letters. Every disk, USB stick, and network share is *mounted* somewhere under a single root `/`. The layout is standardised by the **Filesystem Hierarchy Standard**.

![Filesystem Hierarchy Standard](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/basics/fig1_directory_tree.png)

The two directories you will visit most are `/etc` (configuration) and `/var/log` (logs). Memorise this rule of thumb:

- **Service won't start?** -> `/var/log/<service>/` or `journalctl -u <service>`.
- **Wrong configuration?** -> `/etc/<service>/`.
- **Disk full?** -> `du -sh /* 2>/dev/null | sort -h` to find the biggest top-level directory.
- **Need a binary?** -> `which <cmd>`; it usually lives in `/usr/bin` or `/usr/local/bin`.

A few subtleties worth knowing on day one:

- `/root` is the home of `root`, *not* `/home/root`. Regular users live under `/home`.
- `/proc` and `/sys` are virtual -- they exist only in RAM, exposing kernel state. `du -sh /proc` is meaningless.
- `/tmp` is wiped on most distros at boot (and often mounted as `tmpfs` in RAM). Don't store anything you care about there.

# The Anatomy of a Command

Every Linux command line is parsed by the shell into three kinds of tokens, separated by spaces.

![Anatomy of a shell command](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/basics/fig2_command_anatomy.png)

```
$ ls -l -a -h --color=auto /var/log /etc
  ^^ ^^^^^^^^ ^^^^^^^^^^^^^ ^^^^^^^^^^^^^
  |    |          |             |
  |    |          |             arguments (what to operate on)
  |    |          long option (verbose, --name=value)
  |    short options (single-letter flags, can stack as -lah)
  command (the program to run)
```

Three habits will save you hours:

- `-lah` means the same as `-l -a -h`. Short flags stack.
- Quote anything containing spaces or shell-special characters: `ls "/var/log my dir"`.
- When unsure, ask the command itself: `<cmd> --help` for a quick summary, `man <cmd>` for the full manual (use `/` to search inside, `q` to quit).

# Your First Ten Minutes on a New Server

You have just SSH'd in. Before anything else, answer five questions.

## 1. Who am I, and what can I do?

The shell prompt tells you. The trailing character is the giveaway:

```bash
root@web01:~#       # '#'  -> you are root, unrestricted
alice@web01:~$      # '$'  -> a regular user, sudo if needed
```

**Don't log in as `root` for routine work.** Use a regular account and call `sudo` when you need power. The audit log (`/var/log/auth.log` or `/var/log/secure`) records every `sudo` invocation with a username -- if everyone shares `root`, that trace is gone.

## 2. Where am I, and what's around me?

```bash
$ pwd
/home/alice
$ ls -lah
total 32K
drwxr-xr-x  4 alice alice 4.0K Jan 15 09:30 .
drwxr-xr-x  6 root  root  4.0K Jan 15 09:00 ..
-rw-------  1 alice alice  220 Jan 15 09:00 .bash_history
-rw-r--r--  1 alice alice 3.8K Jan 15 09:00 .bashrc
drwx------  2 alice alice 4.0K Jan 15 09:30 .ssh
drwxr-xr-x  2 alice alice 4.0K Jan 15 09:30 projects
```

`pwd` (print working directory) and `ls -lah` (long, all-including-hidden, human-readable sizes) together tell you almost everything you need.

## 3. What is the system?

```bash
$ uname -a
Linux web01 5.15.0-92-generic #102-Ubuntu SMP x86_64 GNU/Linux
$ cat /etc/os-release | head -3
PRETTY_NAME="Ubuntu 22.04.3 LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
$ hostnamectl
   Static hostname: web01
           Chassis: vm
        Machine ID: 9e2f...
   Operating System: Ubuntu 22.04.3 LTS
            Kernel: Linux 5.15.0-92-generic
      Architecture: x86-64
```

## 4. Are we under pressure?

```bash
$ df -h /                # disk free, root partition
Filesystem      Size  Used Avail Use% Mounted on
/dev/vda1        40G   12G   26G  31% /
$ free -h                # memory + swap
               total        used        free      shared
Mem:           7.8Gi       1.2Gi       3.4Gi       128Mi
Swap:          2.0Gi          0B       2.0Gi
$ uptime
 09:32:11 up 47 days,  3:14,  2 users,  load average: 0.18, 0.22, 0.20
```

Load averages are 1-, 5-, and 15-minute averages of *runnable + uninterruptible* tasks. On a 4-core box, anything sustained above 4.0 means CPU saturation.

## 5. Is the network alive?

```bash
$ ip -br addr            # brief view of all interfaces
lo               UNKNOWN        127.0.0.1/8 ::1/128
eth0             UP             10.0.0.42/24
$ ip route | head -2
default via 10.0.0.1 dev eth0
10.0.0.0/24 dev eth0 proto kernel scope link src 10.0.0.42
$ ping -c 3 1.1.1.1
64 bytes from 1.1.1.1: icmp_seq=1 ttl=58 time=2.14 ms
```

`ip` replaced the deprecated `ifconfig`/`route`/`arp` trio years ago. If you still see `ifconfig` in tutorials, it works on most distros via a compatibility package, but `ip` is what you should learn.

# File and Directory Operations: The Daily Vocabulary

These are the commands your fingers should type without looking. The cheat-sheet below groups them by intent.

![Essential commands by category](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/basics/fig3_command_cheatsheet.png)

## Navigation

```bash
pwd                 # print working directory
cd /var/log         # absolute path
cd projects/api     # relative path
cd ~                # home directory (same as `cd` with no args)
cd ..               # one level up
cd -                # the directory you were just in (toggle)
```

A trick worth knowing: **`pushd` / `popd` / `dirs`** maintain a stack of directories, so you can dive into a deep path and pop back with one keystroke. Tab-completion works on every path.

## Listing

```bash
ls                  # plain listing
ls -l               # long format (perms, owner, size, mtime)
ls -lh              # human-readable sizes (1.4K, 2.3M, 7G)
ls -la              # include hidden files (.bashrc, .ssh)
ls -lah             # all of the above (the one you'll type forever)
ls -lt              # sort by modification time, newest first
ls -lS              # sort by size, largest first
```

`ll` is an alias for `ls -alF` shipped with Bash on Ubuntu/Debian, and many people add their own (`alias ll='ls -lah'` in `~/.bashrc`).

## Creating and Deleting

```bash
mkdir reports                    # one directory
mkdir -p data/2024/01/raw        # parents auto-created
touch notes.md                   # empty file, or update mtime if it exists
rm notes.md                      # delete one file
rm -r reports                    # recursive (directories)
rm -rf reports                   # recursive + force, no prompts
```

**`rm` does not have a trash bin.** Once a file is unlinked, recovery requires forensic tools and even then is unreliable. Two safety habits:

- Run `ls <pattern>` first to *see* what `rm <pattern>` would touch.
- Add `alias rm='rm -i'` to your shell rc if you're cautious -- it asks before deleting each file.
- Never run `rm -rf "$VAR/"` unless you're certain `$VAR` is set; an empty `$VAR` resolves to `rm -rf /`. Modern GNU `rm` refuses `/` by default (`--preserve-root`), but don't rely on it.

## Copying and Moving

```bash
cp report.md report.bak          # copy file
cp -r src/ dst/                  # recursive (directories)
cp -a src/ dst/                  # archive: preserve perms, links, times
cp -i src dst                    # interactive: prompt on overwrite
mv old.md new.md                 # rename in place
mv *.log /var/log/archive/       # move several files
```

`mv` within the same filesystem is just a rename of the directory entry -- effectively free, no data is copied. Across filesystems it's `cp` followed by `rm`.

## Viewing File Contents

```bash
cat short.txt                    # dump entire file (small files only)
less /var/log/syslog             # scrollable pager: space=down, b=up, /=search, q=quit
head -n 20 access.log            # first 20 lines (default 10)
tail -n 50 access.log            # last 50 lines
tail -f /var/log/nginx/access.log    # follow new appends in real time
tail -F /var/log/nginx/access.log    # like -f, but survives log rotation
```

`tail -F` is the standard incantation for watching a log while waiting for a request to come in. Pair it with `grep --line-buffered ERROR` to filter live.

## Quick Edits Without an Editor

```bash
echo "deployed at $(date)" > deploy.log    # overwrite
echo "step 2 done"        >> deploy.log    # append

cat > config.yaml <<'EOF'
host: 0.0.0.0
port: 8080
debug: false
EOF
```

The `<<'EOF'` heredoc with the quoted delimiter prevents shell expansion -- write `$VAR` literally, not its value. Without quotes, variables get substituted.

## Finding Files and Text

Two commands cover ninety percent of needs:

```bash
# By name, type, time, size:
find /var/log -name "*.log"             # by glob (note the quotes)
find /home/alice -type f -mtime -1      # files modified < 1 day
find / -size +100M -type f 2>/dev/null  # files larger than 100 MB
find . -name "*.tmp" -delete            # find AND delete (be careful)

# By contents:
grep "ERROR" app.log                    # one file
grep -rIn "TODO" .                      # recursive, line numbers, skip binaries
grep -v INFO app.log                    # invert: lines NOT containing INFO
grep -E "^(WARN|ERROR)" app.log         # extended regex
```

`-rIn` is the trio you'll memorise: **r**ecursive, skip b**I**nary files (capital I), show line **n**umbers. For very large repositories, `ripgrep` (`rg`) is a faster drop-in.

## File Information

```bash
stat report.md           # full metadata: atime, mtime, ctime, inode, blocks
file mystery.bin         # detect type by content, not extension
wc -l access.log         # count lines  (-w words, -c bytes)
du -sh ./*               # disk usage of every immediate child, summarised
du -sh /var/lib/* 2>/dev/null | sort -h   # ranked from smallest to biggest
```

`stat` distinguishes three timestamps that confuse beginners: **atime** (last read), **mtime** (last content change), **ctime** (last metadata change, e.g. `chmod`). `ls -l` shows `mtime` by default.

# Pipelines: Composing Commands

A pipe `|` connects the **stdout** of one process to the **stdin** of the next. No intermediate file is created; data streams in chunks through kernel buffers. This is the single most powerful idea in the Unix toolchain.

![Pipeline data flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/basics/fig4_pipeline_flow.png)

A practical example: find the top 10 IPs hitting your web server with errors.

```bash
$ cat /var/log/nginx/access.log \
    | grep ' 5[0-9][0-9] '          \   # 5xx HTTP responses
    | awk '{print $1}'              \   # extract IP (column 1)
    | sort                          \   # group identical lines
    | uniq -c                       \   # count consecutive duplicates
    | sort -rn                      \   # numeric, reverse
    | head -10                          # top 10
   1842 203.0.113.42
    611 198.51.100.7
    ...
```

Three companion operators you will reach for constantly:

```bash
cmd > file        # redirect stdout, OVERWRITE
cmd >> file       # redirect stdout, APPEND
cmd 2> errors     # redirect stderr only
cmd > all 2>&1    # merge stderr INTO stdout, then redirect
cmd | tee file    # write stdout to file AND keep streaming downstream
```

`2>&1` is read as "send file descriptor 2 to wherever fd 1 points right now." Order matters: `cmd 2>&1 > file` does *not* capture stderr to the file; `cmd > file 2>&1` does.

> Deeper coverage of pipes, process substitution `<(...)`, named pipes (FIFOs), and `xargs`-driven parallel execution is in the **Linux Advanced File Operations** article.

# SSH: Logging Into a Remote Box

SSH (Secure Shell) is the default way to drive any remote Linux machine. It encrypts everything, runs on TCP port `22` by default, and supports both password and public-key authentication.

## Basic Usage

```bash
ssh alice@10.0.0.42                  # connect on default port 22
ssh -p 2222 alice@example.com        # custom port
ssh -i ~/.ssh/work_ed25519 alice@server   # explicit private key
exit                                 # leave the session (Ctrl-D also works)
```

If your network drops mid-session, the terminal hangs. Type `~.` (tilde, dot) on a fresh line to force-disconnect from the local end.

## Password-Free Login With a Key

The standard, vastly more secure setup. Generate one keypair per laptop and copy the **public** half to every server you visit.

```bash
# 1. On your laptop -- create the key (Ed25519 is current best practice)
ssh-keygen -t ed25519 -C "alice@laptop"
#   -> ~/.ssh/id_ed25519       (private, never share, mode 600)
#   -> ~/.ssh/id_ed25519.pub   (public, safe to copy around)

# 2. Push the public key to the server
ssh-copy-id alice@10.0.0.42
#   appends to ~alice/.ssh/authorized_keys with correct permissions

# 3. Future logins skip the password
ssh alice@10.0.0.42
```

Once keys work, lock down `sshd` by setting `PasswordAuthentication no` in `/etc/ssh/sshd_config` and reloading the service. Bots scan the entire IPv4 space looking for password-accepting `sshd` -- close that door.

## Hardening Checklist

- Use **Ed25519** keys (faster, smaller, modern) instead of 4096-bit RSA when both ends are recent.
- Disable password auth and root login (`PermitRootLogin no`).
- Run `fail2ban` to auto-ban IPs that fail repeatedly.
- Limit SSH access at the firewall to known source CIDRs when possible.
- Changing the port from 22 only reduces noise from drive-by scanners; it is not real security.

> Service management (reloading `sshd`, enabling on boot) is covered in the **Linux System Service Management** article.

# Permissions and Users in Two Pages

## Reading and Changing Permissions

You already know the rwx-triplet shape. Two ways to write a `chmod` change:

```bash
# Numeric: r=4, w=2, x=1, sum per triplet
chmod 755 deploy.sh        # owner=rwx, group=r-x, others=r-x
chmod 600 ~/.ssh/id_ed25519  # owner=rw-, others=---  (required for SSH keys)
chmod 644 README.md        # owner=rw-, group=r--, others=r--

# Symbolic: who (u/g/o/a) +/-/= what (r/w/x)
chmod u+x deploy.sh        # add execute for owner
chmod g-w report.md        # remove write from group
chmod o=  secrets.env      # remove ALL access for others
chmod -R a+r docs/         # recursive, all users get read
```

Common modes worth memorising:

| Mode  | Meaning                | Typical use                   |
| ----- | ---------------------- | ----------------------------- |
| `755` | rwxr-xr-x              | Executables, public scripts   |
| `644` | rw-r--r--              | Normal config / source files  |
| `700` | rwx------              | Private directory             |
| `600` | rw-------              | SSH private key, secrets file |
| `400` | r--------              | Read-only secret              |

> SUID, SGID, sticky bit, ACLs, and `umask` are in the **Linux File Permissions** article.

## Switching User With `su` and `sudo`

```bash
sudo apt update              # run ONE command as root (audited, your password)
sudo -i                      # interactive root shell (audited)
su - root                    # switch to root (requires ROOT password, often disabled)
su - bob                     # switch to bob, load his environment
```

`sudo` is the modern default because it logs *who* did *what* and never requires sharing the root password. On a fresh Ubuntu install, the first user is automatically in the `sudo` group; on RHEL-family systems the equivalent is the `wheel` group.

> User creation, group membership, password policies, `useradd`/`usermod`/`userdel`, and `/etc/passwd`/`/etc/shadow` internals are covered in the **Linux User Management** article.

# Processes, Briefly

```bash
ps aux                            # snapshot of every process
ps aux | grep nginx               # filter by name
ps -ef --forest                   # process tree with parent links
top                               # live, sorted by CPU; press M for memory, q to quit
htop                              # nicer top (often needs install)

kill 1234                         # polite TERM signal -- process can clean up
kill -9 1234                      # KILL -- kernel terminates immediately, no cleanup
killall nginx                     # all processes named nginx
pkill -f 'python worker.py'       # match against full command line

command &                         # run in background of current shell
nohup long-job &                  # survive logout (writes nohup.out)
jobs                              # list backgrounded jobs of this shell
fg %1                             # resume job 1 in the foreground
```

Reach for plain `kill` first; only escalate to `kill -9` if a process is genuinely stuck. `-9` cannot be intercepted, so the program never gets to flush buffers, close sockets, or release locks -- a frequent cause of corrupted state.

> CPU/memory/IO monitoring (`vmstat`, `iostat`, `pidstat`), `nice`/`renice`, control groups, and the OOM killer are in the **Linux Process and Resource Management** article.

# Package Management in 30 Seconds

Linux installs software via a **package manager**, not by downloading installers.

```bash
# Debian / Ubuntu (apt)
sudo apt update                      # refresh package index
sudo apt install nginx
sudo apt remove nginx                # uninstall, keep config
sudo apt purge  nginx                # uninstall + remove config
sudo apt search keyword

# RHEL / CentOS / Rocky / Alma (dnf, or yum on 7)
sudo dnf install nginx
sudo dnf remove  nginx
sudo dnf search  keyword
sudo dnf upgrade                     # update everything

# Arch (pacman)
sudo pacman -Syu                     # sync repos + upgrade system
sudo pacman -S    nginx
sudo pacman -R    nginx
```

Why this pays off: dependencies are resolved automatically, every package is signature-verified by the distro, and one upgrade command covers the whole system.

> Building from source, manual `.deb`/`.rpm` install, alternate ecosystems (`snap`, `flatpak`, `pip`, `npm`), and managing third-party repositories live in the **Linux Package Management** article.

# Habits That Will Save You

1. **Back up before you edit anything in `/etc`.**
   ```bash
   sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.bak.$(date +%F)
   ```
   When the new config breaks SSH, the old file is one `cp` away.

2. **Never modify SSH on a single connection.** Open a *second* SSH session before editing `sshd_config`. If the change locks the new session out, the old one is still alive and can revert. This habit has saved entire weekends.

3. **Treat these commands as nuclear:**
   - `rm -rf /` -- erase the universe (modern `rm` refuses, but don't test it).
   - `dd if=/dev/zero of=/dev/sda` -- zero an entire disk, no recovery.
   - `chmod -R 777 /` -- world-writable everything; system is gone, security-wise.
   - `> /etc/passwd` -- truncate the user database. The system stays up until reboot.

   Always confirm the target *before* the dangerous flag (`ls`, `lsblk`, `cat`).

4. **The logs are usually telling you the answer.** Check them first.
   ```bash
   sudo tail -f /var/log/syslog                    # general (Debian/Ubuntu)
   sudo tail -f /var/log/messages                  # general (RHEL family)
   sudo grep 'Failed password' /var/log/auth.log   # SSH brute-force attempts
   sudo journalctl -u nginx -n 50 --no-pager       # last 50 lines for one service
   ```

# Where to Go Next

This article exists to give you a map. Each topic has its own dedicated piece for when you need depth:

- **Linux File Permissions** -- SUID/SGID/sticky, ACLs, `umask`, inheritance.
- **Linux User Management** -- `useradd`, groups, sudoers syntax, PAM, password policy.
- **Linux Disk Management** -- partitioning, filesystems, mounting, LVM, RAID.
- **Linux System Service Management** -- systemd units, `journalctl`, timers.
- **Linux Package Management** -- repos, GPG, building from source, alternatives.
- **Linux Process and Resource Management** -- monitoring, cgroups, scheduling, OOM.
- **Linux Advanced File Operations** -- pipes, redirection, `xargs`, `tee`, FIFOs.

Linux is not learned in one sitting. Spin up a free-tier VM (or a local VirtualBox / Multipass / WSL2 instance), open a terminal, and break things on purpose -- it's the fastest way for the muscle memory to set in.

# References

- [The Linux Documentation Project](https://tldp.org/) -- canonical, if dated, reference library.
- [Arch Linux Wiki](https://wiki.archlinux.org/) -- the highest signal-to-noise Linux documentation on the internet, useful well beyond Arch.
- [The Linux Command Line, William Shotts](http://linuxcommand.org/tlcl.php) -- a free book-length introduction to the shell.
- [Linux Performance, Brendan Gregg](https://www.brendangregg.com/linuxperf.html) -- the canonical resource page on measuring Linux at scale.
- [Filesystem Hierarchy Standard 3.0](https://refspecs.linuxfoundation.org/FHS_3.0/fhs/index.html) -- the formal spec for the directory layout.
