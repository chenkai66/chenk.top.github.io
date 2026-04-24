---
title: "Linux Process and Resource Management: From `top` to cgroups"
date: 2024-02-02 09:00:00
tags:
  - Linux
  - Cloud
  - System Administration
categories: Linux
series: "Linux"
series_order: 7
series_total: 8
lang: en
mathjax: false
description: "How processes are born and die on Linux: the fork/exec model, the state machine that ps and top print in the STAT column, the resource axes (CPU, memory, disk I/O, network) that actually constrain a server, the signals you should know by heart, and the cgroup + namespace primitives that turn all of this into containers."
---

The job of a Linux operator is rarely "memorise more commands". It is to take a fuzzy symptom — *the site feels slow, the API timed out, the box is unresponsive* — and quickly **map it to the right axis**: is the CPU saturated, is memory being eaten by cache (which is fine) or by a runaway process (which is not), is the disk queue full, is some socket leaking? Once the axis is named, the tool follows almost mechanically.

This post walks the full picture in that order. We start from how a process actually comes to exist (`fork()` + `exec()`), the state machine the kernel pushes it through, and the four resource axes that bound everything it can do. Then we build up the toolchain — `top` / `htop` / `ps` / `pstree` / `lsof` / `ss` / `iostat` — not as a command list but as a layered way of looking at the same system. Finally we cover the things you do *to* a process: signals, background jobs, `nice`/`renice` priorities, and the cgroup + namespace mechanisms that quietly underpin every container you have ever run.

## Process, program, thread — and why the distinction matters

![Process states](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/process-resource-management/fig1_process_states.png)

These three terms are used interchangeably in casual conversation, but the kernel treats them very differently:

| Term        | What it is                                                                | Identifier              |
| ----------- | ------------------------------------------------------------------------- | ----------------------- |
| **Program** | A static file on disk (`/usr/bin/vim`) — instructions plus data layout    | inode + path            |
| **Process** | A running instance of a program: own address space, file descriptors, PID | PID, parent PPID        |
| **Thread**  | An execution flow *inside* a process, sharing its address space           | TID (kernel `task_struct`) |

A process is what you bill resources against — memory pages, file descriptors, open sockets all live in the process. A thread is just another schedulable entity that happens to share that state with its siblings. On Linux this distinction is unusually thin: both processes and threads are `task_struct` to the kernel; they only differ in *what* they share when created (`clone()` flags decide). That is why `ps -eLf` lists threads with the same PID but different LWP values, and why a multi-threaded JVM shows up as one process with hundreds of `task_struct`s underneath.

A few invariants worth keeping in mind:

- Every process has a **parent**. PPID 0 is reserved for the kernel; PPID 1 is `systemd` (or `init`), the only userspace process the kernel itself starts. Walk PPIDs up far enough and you always hit 1.
- Processes are **isolated by default**: address spaces don't overlap, opening a file in process A does nothing to process B. Sharing requires explicit IPC (pipes, sockets, shared memory, signals).
- Processes are **dynamic**: they are created, scheduled, blocked, woken, and torn down constantly. A healthy server churns thousands of short-lived processes per minute; that is normal.

## How processes are born: fork() and exec()

![fork/exec model](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/process-resource-management/fig2_fork_exec_model.png)

Linux has exactly one mechanism for creating a new process: **fork**. `fork()` (or its modern cousin `clone()`) duplicates the calling process. Right after the call you have two near-identical processes returning from the same line of code; the only difference is the return value (`0` in the child, the child's PID in the parent). They share open file descriptors, the same code, the same heap contents — initially even the same memory pages, thanks to copy-on-write.

That alone would just multiply the same program. The second half of the trick is **exec**. Inside the freshly forked child, calling `execve("/bin/ls", argv, envp)` tells the kernel to *replace* the current program image with a different binary while keeping the same PID, the same PPID, and (by default) the same file descriptors. So when you type `ls` at a shell prompt, what actually happens is:

1. The shell calls `fork()`. Now there are two shells.
2. The child calls `execve("/bin/ls", ...)`. Its program image becomes `ls`.
3. The parent (the original shell) calls `waitpid()` to block until the child exits, then collects the exit code into `$?`.

This split between "create a process" and "load a program" is what makes shell pipelines, redirection, environment manipulation, and `setuid`-style privilege drops possible. Each of those happens *between* fork and exec, in the child, before the new program even starts.

Two consequences of this design show up constantly:

- **Inherited file descriptors.** A child inherits all open FDs of the parent unless they are marked `O_CLOEXEC`. That is why `cmd > log.txt` works: the shell opens the file as FD 1 in the child *before* exec-ing the command. It is also why a leaking daemon can pin deleted files (we'll come back to that).
- **PID 1 is special.** If a process exits while it still has children, those orphans get re-parented to PID 1, which is responsible for `wait()`-ing on them. A container's PID 1 has the same job — and if you forget to handle it, you get the famous "zombies in my Docker container" problem.

## The process state machine

`ps` and `top` show a one-letter `STAT` column for every process. Those letters correspond to the state the kernel currently has the task in:

| Code | State                | Meaning                                                     |
| ---- | -------------------- | ----------------------------------------------------------- |
| `R`  | Running              | On a CPU right now, or on the runqueue waiting for one      |
| `S`  | Interruptible sleep  | Waiting for an event (read from socket, timer); woken by signals |
| `D`  | Uninterruptible sleep| Waiting on something the kernel won't let go of, usually disk I/O — even `kill -9` won't move it |
| `T`  | Stopped              | Paused by `SIGSTOP`/`SIGTSTP` (e.g. you hit Ctrl+Z)          |
| `Z`  | Zombie               | Already exited, but parent hasn't called `wait()` to reap it |
| `X`  | Dead                 | Transient state during teardown, you almost never see it     |

The state is not just trivia. A box with 200 processes in `D` is not "100 % CPU bound" — it is *blocked on storage*, and no amount of CPU tuning will help. A growing pile of `Z` processes points at a buggy parent that forgot to `wait()`. Stuck `T` processes mean somebody (or some script) sent `SIGSTOP` and never followed up with `SIGCONT`.

`top`'s top line — `Tasks: 150 total, 2 running, 148 sleeping, 0 stopped, 0 zombie` — is your first sanity check. Anything other than zero in `stopped` or `zombie` deserves a follow-up.

## The four resource axes

Before drowning in tools, fix the mental model. Almost every production problem maps to one of four axes:

1. **CPU** — how much computation per second can be performed.
2. **Memory** — how much working set the process can hold without going to disk.
3. **Disk I/O** — how fast bytes can flow to and from persistent storage.
4. **Network** — how fast bytes can flow in and out of the box.

A bottleneck on any of them slows everything that depends on it. The trick is figuring out *which* one — and that's mostly what monitoring tools are for.

### CPU

What you usually want to know:

- How many cores does the box actually have? `nproc` and `lscpu` answer that.
- How loaded is it? `uptime` shows three load averages: 1, 5 and 15 minutes.

Load average is *not* CPU utilisation. It counts processes that are either running or **runnable plus uninterruptible** (`R + D`). A 4-core box with `load avg 4.0` and 95 % idle CPU is almost always a disk problem: tasks are piling up in `D` waiting for I/O, not for CPU. Read load against the number of cores: `< cores` is fine, `≈ cores` is full, `>> cores` is overloaded.

```bash
uptime
# 06:56:12 up 12 days,  3:45,  3 users,  load average: 0.22, 0.45, 0.56
nproc
# 4
```

### Memory

`free -h` is the canonical view. The trap is reading it like Windows Task Manager:

```text
              total        used        free      shared  buff/cache   available
Mem:           15Gi       2.5Gi       1.0Gi       100Mi       11.5Gi        12Gi
Swap:         2.0Gi          0B       2.0Gi
```

Newcomers see `free: 1.0Gi` and panic. **Don't.** Linux deliberately uses spare RAM as **buffer/cache** to speed up I/O — that 11.5 GiB is reclaimable on demand. The number that matters is `available`: it accounts for the cache the kernel believes it could free if a process asked for memory.

The two flavours of cached memory:

- **Buffer** is a write-side staging area. When you write a file, bytes land in the page cache and are flushed to disk in batches; that is what makes small writes fast.
- **Cache** is the read-side: pages from disk that the kernel keeps around in case they are read again, which they very often are.

You are *actually* low on memory when **all three** of these become true at once:

- `available` is near zero,
- `swap used` is climbing rather than just non-zero,
- the kernel starts logging OOM kills (`dmesg | grep -i 'killed process'`).

### Disk and network

Disk capacity is `df -h` (per filesystem) and `du -sh *` (per directory). Real-time I/O lives in `iostat -x 1` and `iotop`; the two metrics that matter are `%util` (how busy the device is) and `await` (how long requests sit in the queue). A device pegged at `%util 100` with `await` in the tens of milliseconds is a real bottleneck.

Network is `ss -tulnp` for who is listening, `ip -s link` for interface counters and drops, and `iftop` (or `nload`) for live bandwidth. For "who is using port 80" the fastest answer is `lsof -i :80` or `ss -tulnp | grep :80`.

## The monitoring toolchain

### `top` — the first thing to run

![top dissected](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/process-resource-management/fig3_top_dissected.png)

`top` is the cockpit view. Every region tells you something specific:

- **Header:** uptime, users, load average. Compare load against `nproc`.
- **Tasks:** total / running / sleeping / **stopped / zombie**. Last two should be zero.
- **`%Cpu(s)`:** broken down into user (`us`), system (`sy`), nice (`ni`), idle (`id`), I/O wait (`wa`), hardware/software interrupts (`hi`/`si`), and stolen (`st`). High `wa` means slow disk. **High `st` on a VM means the hypervisor is giving your CPU to other tenants** — common in noisy-neighbour clouds.
- **Mem / Swap:** trust `avail Mem`, not `free`. If `Swap used` is climbing, the box is under real memory pressure.
- **Process table:** sortable by hotkey. `P` sorts by `%CPU`, `M` by `%MEM`, `T` by time. `1` toggles per-CPU breakdown — invaluable when one core is pinned and the others sit idle. `k` prompts for a PID and signal. `q` quits.

### `htop` — top with a UI

`htop` is `top` with colour, mouse support, a tree view (`F5`) and a built-in signal sender (`F9`). On any box where you spend more than a minute in `top`, install it:

```bash
sudo apt install htop          # Debian/Ubuntu
sudo dnf install htop          # RHEL/Rocky/Fedora
```

### `ps` — the static snapshot

`top` refreshes; `ps` freezes a moment in time, which is often what you want for scripting and grep'ing.

```bash
ps -ef                  # System V style: every process, full format
ps aux                  # BSD style: every process including those without a TTY
ps -eo pid,ppid,user,stat,%cpu,%mem,cmd --sort=-%cpu | head -10
```

The columns worth knowing (`ps aux`):

- `VSZ` — virtual memory size (what the process *asked for*; mostly meaningless on its own).
- `RSS` — resident set size, **actual physical pages**. This is the honest memory number.
- `STAT` — the state codes from the table above, sometimes with extra suffixes (`<` high-priority, `N` low-priority, `s` session leader, `+` foreground process group).
- `TIME` — accumulated CPU time, not wall-clock.

### `pstree` — who spawned whom

```bash
pstree -ap            # show command line + PIDs
pstree -ap <PID>      # subtree under a single PID
```

When a misbehaving process keeps coming back, look up the tree: something is respawning it.

### `lsof` — every open file is something

`lsof` lists open files, where "file" is the Unix everything-is-a-file sense — regular files, sockets, pipes, devices, even kernel-side handles.

```bash
lsof -p <PID>          # files this PID has open
lsof -c nginx          # files any nginx process has open
lsof -u alice          # files owned by alice's processes
lsof -i :80            # who is bound to port 80
lsof -i tcp            # all TCP sockets
lsof +D /var/log       # everything open under /var/log
lsof +L1               # files with link count < 1 — deleted but still pinned
```

The `FD` column is a tiny grammar of its own:

- `cwd` — the process's current working directory
- `txt` — the program binary itself
- `mem` — memory-mapped file (shared library)
- `0r`, `1w`, `2w` — stdin / stdout / stderr
- `Nu` (`u`/`r`/`w`) — a regular open FD with mode

### `ss` — sockets, the modern `netstat`

```bash
ss -tulnp            # tcp + udp + listening + numeric + with PID
ss -s                # one-line summary of all socket states
ss -tan state established '( dport = :443 )'
```

### `iostat` and `iotop` — the disk axis

```bash
iostat -x 1          # per-device extended stats, refresh every second
sudo iotop -o        # only processes currently doing I/O
```

Watch `%util`, `r/s`, `w/s`, `await`. A device at `%util 100` with single-digit `await` is *busy but healthy*; one at `%util 100` with `await` in the hundreds is *queued and miserable*.

## Controlling processes

### Signals

![Signals reference](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/process-resource-management/fig5_signals_table.png)

`kill` is misnamed. It does not kill anything; it **sends a signal**, and the signal's *default* action happens to be "terminate" for most signals. The handful that actually matter day to day:

| Signal    | Number | What it does                                                       |
| --------- | ------ | ------------------------------------------------------------------ |
| `SIGTERM` | 15     | Polite request to terminate; process can clean up. **Always try this first.** |
| `SIGINT`  | 2      | What Ctrl+C sends; semantically "the user interrupted you"        |
| `SIGHUP`  | 1      | Originally "the terminal hung up"; by convention, daemons treat it as **reload config** (nginx, sshd, syslog) |
| `SIGKILL` | 9      | The nuclear option. Kernel kills the process, no cleanup, **uncatchable**. Last resort. |
| `SIGSTOP` | 19     | Pause the process. **Uncatchable.** Resume with `SIGCONT`.        |
| `SIGTSTP` | 20     | Catchable variant of stop — what Ctrl+Z sends                     |
| `SIGCONT` | 18     | Resume a stopped process                                          |
| `SIGUSR1` / `SIGUSR2` | 10 / 12 | Application-defined. Many daemons use `SIGUSR1` to rotate logs. |
| `SIGCHLD` | 17     | Sent to the parent when a child changes state. The parent should `wait()`. |
| `SIGPIPE` | 13     | You wrote to a pipe with no reader (`yes | head -1`).             |

The right escalation is **always**: try `SIGTERM`, give the process a few seconds, only then escalate to `SIGKILL`. `kill -9` skips destructors, leaves locks held, leaves temp files behind, and corrupts databases that were mid-write. Reach for it last.

```bash
kill <PID>             # SIGTERM (the default)
kill -HUP <PID>        # reload config
kill -9 <PID>          # SIGKILL — last resort
kill -l                # list every signal name and number on this kernel
pkill -HUP nginx       # by name and pattern
killall -USR1 nginx    # by exact program name
```

### Background jobs and detachment

When you start something that needs to outlive your SSH session, three options in increasing order of robustness:

```bash
./long_task.sh &                                  # backgrounds in this shell only;
                                                  # dies on SIGHUP when SSH drops
nohup ./long_task.sh >/dev/null 2>&1 &           # ignore SIGHUP; output discarded
tmux new -s work     # then run it inside; detach with Ctrl-B d, reattach later
```

`tmux` (or the older `screen`) is the right answer for anything interactive or long-running. For *services*, none of these are appropriate — write a `systemd` unit and let the init system supervise it (covered in the service-management post).

Foreground/background within one shell:

```bash
./task.sh         # foreground
^Z                # SIGTSTP — pauses, returns to shell
bg %1             # resume in background
fg %1             # bring back to foreground
jobs              # list jobs in this shell
disown %1         # detach from shell, survive logout
```

### Priorities: `nice` and `renice`

The scheduler picks who runs next partly based on a process's **nice value**, an integer from `-20` (highest priority, scheduler favours it) to `19` (lowest, polite to others). Default is `0`. Lowering nice below 0 requires root.

```bash
nice -n 19  ./backup.sh                # start a low-priority background backup
nice -n -10 ./important.sh             # higher priority (root)
renice -n 10 -p <PID>                  # change nice on a running process
renice -n 5 -u alice                   # all of alice's processes
```

`nice` is a soft hint — it influences scheduling weight but a low-priority process still gets CPU when nothing else wants it. For *hard* limits (e.g. "this batch job may use no more than 2 cores and 4 GiB"), use cgroups, not nice.

### Orphans and zombies

Two states that confuse newcomers, both rooted in the parent–child contract:

- **Orphan**: parent exits before the child. The kernel re-parents the child to PID 1, which inherits the duty to `wait()` on it. **Harmless** in normal operation.
- **Zombie (`Z`)**: child has already exited, but the parent has not called `wait()` to collect its exit status. The `task_struct` lingers in the process table holding a PID and exit code. Zombies use no CPU or memory, but they consume process-table slots — accumulate enough and `fork()` itself starts to fail.

The fix for zombies is to fix the parent. If the parent is a daemon under your control, it must `wait()` (or set up a `SIGCHLD` handler, or set `SIGCHLD` to `SIG_IGN` so the kernel auto-reaps). If the parent is third-party and broken, killing the parent re-parents the zombies to `systemd`, which reaps them immediately. Rebooting is the answer of last resort.

You can list current zombies cheaply:

```bash
ps -eo pid,ppid,stat,cmd | awk '$3 ~ /^Z/'
```

## cgroups + namespaces — the foundation under containers

![cgroups and namespaces](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/process-resource-management/fig4_cgroups_namespaces.png)

Once you understand processes, two kernel features explain how containers actually work — and they are useful even if you never touch Docker.

**Namespaces** isolate *what a process can see*. Each namespace virtualises a different system resource:

- `pid` — own PID space; PID 1 inside is not PID 1 outside.
- `net` — own network interfaces, routing table, sockets.
- `mnt` — own mount table; the root filesystem can be entirely different.
- `uts` — own hostname and domain name.
- `ipc` — own SysV/POSIX IPC objects.
- `user` — own UID/GID range; root inside maps to an unprivileged UID outside.
- `cgroup` — own view of the cgroup hierarchy.

**cgroups (control groups, v2)** limit *what a process can use*. A cgroup is a directory under `/sys/fs/cgroup/` containing the PIDs that belong to it and a set of files that configure resource caps:

```bash
# create a group, cap it at half a CPU and 512 MiB, run a shell in it
sudo mkdir /sys/fs/cgroup/demo
echo "50000 100000" | sudo tee /sys/fs/cgroup/demo/cpu.max     # 50ms per 100ms
echo $((512*1024*1024)) | sudo tee /sys/fs/cgroup/demo/memory.max
echo $$ | sudo tee /sys/fs/cgroup/demo/cgroup.procs            # add this shell
```

The controllers worth knowing:

- `cpu.max` — bandwidth cap (quota / period).
- `cpu.weight` — relative share when contended.
- `memory.max` — hard ceiling; exceeding it triggers the OOM killer **inside the cgroup** rather than across the host.
- `io.max` — per-device IOPS and BPS caps.
- `pids.max` — fork-bomb protection; cap the number of processes in the group.
- `cpuset.cpus` — pin to specific CPUs / NUMA nodes.

A container engine like Docker, Podman or `containerd` is, fundamentally, a small program that:

1. Creates a fresh set of namespaces (`unshare(2)` or `clone(2)` with the right flags).
2. Pivots into a new root filesystem extracted from an image.
3. Puts the resulting process tree into a cgroup with the caps you asked for (`docker run --cpus 0.5 --memory 512m`).

Once you have seen this, container debugging becomes much less mysterious. `nsenter -t <PID> -a` enters a running container's namespaces; `cat /proc/<PID>/cgroup` tells you which group it lives in; `systemd-cgtop` is `top` for cgroups.

## Walkthrough: "the box feels slow, what now?"

A repeatable order of operations for triaging "slow":

```bash
# 1. Is anything obviously on fire?
uptime                          # load average vs. nproc
dmesg -T | tail -50             # OOM kills, hardware errors, segfaults
df -h                           # any filesystem near 100 %?

# 2. Which axis is saturated?
top                             # press 1 for per-CPU; watch us/sy/wa/st
free -h                         # available memory? swap growing?
iostat -x 1 5                   # %util and await per device
ss -s                           # socket counts; lots of TIME-WAIT?

# 3. Who is doing it?
ps aux --sort=-%cpu | head      # top CPU consumers
ps aux --sort=-%mem | head      # top RSS consumers
sudo iotop -o                   # who is reading/writing
sudo lsof -i -nP | head         # who is opening sockets

# 4. Drill into a specific PID
cat /proc/<PID>/status          # state, threads, memory totals
ls -l /proc/<PID>/fd/           # what files are open
cat /proc/<PID>/limits          # ulimits in effect for that process

# 5. Decide and act
renice -n 10 -p <PID>           # de-prioritise a runaway batch job
kill <PID>                      # ask politely first
kill -9 <PID>                   # only after SIGTERM is ignored
```

Resist the temptation to skip step 2. Most "the server is slow" tickets are misdiagnosed because somebody jumped straight to `top`, saw 100 % CPU on one process, and `kill -9`'d it — when the real problem was a saturated disk and the process was simply waiting in `D`.

## Real-world: recovering an accidentally deleted log file

Classic incident: someone runs `rm -rf /var/log/nginx/access.log` while nginx is happily writing to it. After the `rm`:

- `ls /var/log/nginx/` no longer shows the file (the directory entry is gone).
- `df -h` reports the same usage as before (the inode and data blocks are still allocated).
- nginx keeps writing to **the same FD** as if nothing happened — because it is.

This is straight out of Unix semantics: a file is unlinked from its directory, but as long as any process still holds an open FD to it, the inode lives on. The kernel exposes that FD under `/proc/<pid>/fd/<fd>`, which means you can copy the file back out:

```bash
PID=$(pidof nginx | awk '{print $1}')

# 1. Confirm the deleted file is still open
sudo lsof -p "$PID" | grep access.log
# nginx 1234 root  6w  REG  8,1  123456789  4711  /var/log/nginx/access.log (deleted)

# 2. Recover by copying out of /proc
sudo cp /proc/"$PID"/fd/6 /var/log/nginx/access.log

# 3. Tell nginx to reopen its log files
sudo nginx -s reload    # or: sudo kill -USR1 "$PID"
```

The same trick works for any deleted-but-still-open file. The moment the last FD is closed, though, the kernel really frees the inode — so do this *now*, before the daemon restarts.

## Recap

- A **process** is a running program with its own address space; **threads** share that space; only `task_struct`s exist as far as the scheduler is concerned.
- New processes are created by **`fork()` + `exec()`**, never from scratch. PID 1 (`systemd` / `init`) is the root of every process tree.
- The **state machine** (`R`/`S`/`D`/`T`/`Z`) is what `ps`/`top` print, and each state implies a different troubleshooting path.
- Resource bottlenecks live on four axes — **CPU, memory, disk I/O, network** — and most monitoring tools are just different views of the same axes.
- Linux memory looks scarier than it is: **`available`, not `free`**, is the number to trust.
- `kill` sends **signals**; `SIGTERM` first, `SIGKILL` last. Daemons reload on `SIGHUP`.
- `nice`/`renice` are soft scheduling hints; **cgroups** are hard caps and they are the real foundation of containers, alongside **namespaces**.

### Further reading

- Brendan Gregg, *Linux Performance* — <http://www.brendangregg.com/linuxperf.html>
- `man proc(5)` — exhaustive reference for `/proc`
- `man 7 signal` — every signal, default action, and catchability rule
- `man 7 cgroups` — cgroup v2 design and controller list
- `man 7 namespaces` — what each namespace virtualises

### Up next in this series

- **Linux Disk Management** — partitions, filesystems, LVM and the mount stack
- **Linux User Management** — users, groups, sudo, PAM, and the principle of least privilege

By this point you should be able to walk into a slow box, name the saturated axis within a minute, and pick a tool with intent rather than by reflex. That is the difference between *running `top`* and *operating a system*.
