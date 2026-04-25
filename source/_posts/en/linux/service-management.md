---
title: "Linux Service Management: systemd, systemctl, and journald"
date: 2022-03-07 09:00:00
tags:
  - Linux
  - Cloud
  - System Administration
categories: Linux
series: "Linux"
series_order: 6
series_total: 8
lang: en
mathjax: false
description: "A working model of systemd: PID 1, units and targets, the service lifecycle, writing your own unit files, journalctl filtering, timers as a cron replacement, and a disciplined troubleshooting workflow."
disableNunjucks: true
---

A "service" on Linux is a long-running background process whose
job is to be there when something needs it: synchronise the clock,
listen for SSH connections, accept HTTP requests, run a backup at 3 AM.
You almost never start one of these by hand. Something has to start
them at boot, restart them when they crash, capture their logs, decide
what depends on what, and shut everything down cleanly when the machine
powers off. On every modern distribution that something is
**systemd**.

This article is the working model I wish someone had handed me when I
first got dropped into a production server. We start from why a
dedicated service manager exists at all, build up the unit/target
mental model, walk through the `systemctl` commands you actually use
day to day, dissect a unit file line by line, and then cover the
adjacent surfaces: `journalctl` for logs, `.timer` units as the modern
cron, and a step-by-step playbook for "the service won't start."

# 1. Why a service manager exists

## The problem before systemd

A bare Linux kernel knows how to load drivers, mount the root
filesystem, and execute exactly one program (PID 1). Everything else
is somebody else's problem. Historically that "somebody" was
**SysV init**: a small program that read scripts out of `/etc/init.d/`
and ran them, in alphabetical order, one after another. Each script
was a hand-rolled shell program responsible for forking the daemon
into the background, writing a PID file, and pretending to know
whether the service was actually up.

That approach worked, but it accumulated three structural problems:

1. **Serial startup is slow.** Running 80 init scripts sequentially
   on a modern server wastes most of a multi-core CPU just waiting on
   I/O.
2. **Dependencies live in the scripts themselves.** If `nginx` needs
   the network up first, you encode that by hoping the alphabet
   cooperates (`S20network` before `S80nginx`) and by sprinkling
   `sleep` calls through the scripts. It is exactly as fragile as it
   sounds.
3. **There is no shared concept of state.** Each daemon decides for
   itself how to background, where to write its PID, how to log, how
   to restart on crash. The init system has no way to ask "is sshd
   actually running?" beyond "did this script exit 0?"

systemd replaces all of that with a uniform supervisor that knows the
difference between "the start command returned" and "the service is
ready," parallelises whatever it can, and keeps a structured log of
everything that happens.

## The systemd model in one picture

![systemd architecture: PID 1, units, and targets](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/service-management/fig1_systemd_architecture.png)

Three layers, top to bottom:

- **PID 1** — `systemd` itself, started by the kernel as the first
  user-space process. It never exits while the system is running.
- **Units** — every manageable thing on the system is a *unit*: a
  daemon (`.service`), a socket waiting for connections (`.socket`), a
  scheduled job (`.timer`), a mount point (`.mount`), and so on.
  systemd reads unit files, tracks each unit's state, and drives it
  between states.
- **Targets** — synchronisation points that group units together.
  `multi-user.target` is the classic "the box is up and serving"
  state; `graphical.target` adds a desktop on top of it. Targets
  replace the SysV `runlevel` concept with something more flexible
  (any unit can pull in any target).

Concretely, when you type `systemctl restart nginx` you are asking
PID 1 to drive the `nginx.service` unit through its state machine —
stop the current process group, wait for it to exit, run `ExecStart`
again, watch for readiness, update the cached state. Every other
command in this article is a variation on that theme.

## Unit types worth knowing

| Unit type | What it manages | Example |
|---|---|---|
| `.service` | A long-running process (the 90% case) | `sshd.service`, `nginx.service` |
| `.socket` | A listening socket; starts a service on first connection | `sshd.socket`, `docker.socket` |
| `.target` | A named group of units (a synchronisation point) | `multi-user.target`, `network-online.target` |
| `.timer` | A scheduled trigger for another unit (cron replacement) | `logrotate.timer`, `apt-daily.timer` |
| `.mount` | A filesystem mount, derived from `/etc/fstab` or a unit file | `home.mount`, `var-log.mount` |
| `.path` | Watches a file or directory; activates a unit on change | `systemd-tmpfiles-clean.path` |
| `.slice` | A cgroup container used to apply resource limits to a group of units | `system.slice`, `user.slice` |

For most of this article "service" means a `.service` unit. The other
types share the same lifecycle and the same management commands, so
once you understand services the rest comes for free.

# 2. systemctl: the day-to-day commands

`systemctl` is your one entry point. It is worth memorising the
common subcommands until they become muscle memory — you will type
them dozens of times a day on a busy server.

## Start, stop, restart, reload

```bash
sudo systemctl start  nginx       # take effect now, do not persist across reboot
sudo systemctl stop   nginx
sudo systemctl restart nginx      # stop then start (drops connections)
sudo systemctl reload  nginx      # ask the daemon to re-read its config
                                  # (only works if the unit file defines ExecReload)
sudo systemctl status  nginx      # current state + last 10 log lines
```

The split between `restart` and `reload` matters in production.
`restart` always works but tears down whatever the service was doing
mid-flight. `reload` is graceful — for nginx it spawns new workers
with the new config and lets the old ones drain — but only the
service author can implement it. `systemctl cat nginx.service` tells
you whether `ExecReload=` is defined.

## Boot-time enablement

```bash
sudo systemctl enable  nginx      # auto-start at next boot (creates a .wants/ symlink)
sudo systemctl disable nginx      # remove the symlink
sudo systemctl is-enabled nginx   # prints: enabled / disabled / static / masked

sudo systemctl enable  --now nginx   # enable + start in one step
sudo systemctl disable --now nginx   # disable + stop in one step
```

`enable` and `start` are independent. A freshly installed package is
usually started but not enabled — it will run until the next reboot
and then disappear. The `--now` flag fuses the two so you don't
forget.

A subtle one: **`mask`** is "stronger than disable." It points the
unit at `/dev/null`, which makes it impossible to start even by
accident or as a dependency of something else. Use it when you really
want a service gone (`sudo systemctl mask firewalld`); reverse with
`unmask`.

## Listing and inspecting

```bash
systemctl list-units --type=service --state=running     # what's up right now
systemctl list-units --type=service --all               # everything systemd knows about
systemctl list-units --type=service --state=failed      # the troubleshooting starting point
systemctl list-unit-files --type=service                # all units on disk, enabled or not

systemctl cat   sshd.service                            # the unit file (with all drop-ins)
systemctl show  sshd.service                            # every property systemd tracks
systemctl list-dependencies sshd.service                # tree of After=/Requires=/Wants=
```

Two are particularly underused. `systemctl cat` is how you read a
unit file *correctly* — it concatenates the base file in
`/usr/lib/systemd/system/` with any drop-in overrides under
`/etc/systemd/system/<unit>.d/`, which is exactly what systemd itself
sees. And `systemctl list-dependencies` makes the ordering graph
visible, which is invaluable when something refuses to start because
something else hasn't.

# 3. The service lifecycle

Once you have started a service, it lives inside a small state
machine. Every line `systemctl status` prints is just a label on one
of these states.

![Service lifecycle: states reported by systemctl status](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/service-management/fig2_service_lifecycle.png)

- **inactive** — the unit exists but no process is running for it.
- **activating** — `ExecStart=` is in flight. For `Type=simple` this
  state is essentially instantaneous; for `Type=notify` the service
  stays here until it calls `sd_notify(READY=1)`.
- **active** — the service is running. For most services this means
  the main process is alive; for `oneshot` units it means the command
  succeeded.
- **deactivating** — `ExecStop=` is in flight, or systemd is sending
  signals (SIGTERM, then SIGKILL after `TimeoutStopSec`).
- **failed** — the service exited non-zero, was killed by a signal,
  or its watchdog tripped. The unit stays in this state until you
  restart or reset it (`systemctl reset-failed`).

The crucial transition is the dashed loop on the diagram:
`failed → activating`. With `Restart=on-failure` (or `Restart=always`)
systemd treats `failed` as transient, waits `RestartSec` seconds, and
runs `ExecStart=` again. This is the entire reason you don't need
something like Monit on top of systemd — the supervisor is already
there.

# 4. Writing your own service

The path from "I have a script" to "it survives reboots and crashes"
is a single unit file. Suppose you have an HTTP server at
`/usr/local/bin/myapp` and you want it to behave like a real service.

## A minimal unit file

Create `/etc/systemd/system/myapp.service`:

```ini
[Unit]
Description=My Custom Application
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/myapp --port 8080
Restart=on-failure
RestartSec=5
User=myapp
Group=myapp

[Install]
WantedBy=multi-user.target
```

Then make systemd notice it, start it, and enable it:

```bash
sudo systemctl daemon-reload          # re-read unit files from disk
sudo systemctl enable --now myapp     # start now and on every boot
sudo systemctl status myapp
```

That's it. The service now restarts automatically on crash, comes back
after reboot, runs as a non-root user, and its stdout/stderr land in
the journal where you can query them with `journalctl -u myapp`.

## Anatomy of the unit file

![Anatomy of a .service unit file](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/service-management/fig3_unit_file_anatomy.png)

The file has three sections, and each one answers a different
question.

### `[Unit]` — what is this thing and what depends on what

```ini
[Unit]
Description=My Custom Application      # one-line human description
Documentation=https://example.com/docs  # shown in `systemctl status`
After=network-online.target             # ordering: start AFTER this is up
Wants=network-online.target             # weak dependency: pull it in, don't fail if it dies
Requires=postgresql.service             # strong dependency: stop us if it stops
ConditionPathExists=/etc/myapp.conf     # skip activation if the path is missing
```

The pair to understand here is **ordering** vs **requirement**.
`After=` and `Before=` only say *when* — they don't pull anything in.
`Wants=` and `Requires=` only say *what* needs to also be running —
they don't say in which order. You almost always want both:
`Wants=network-online.target` *and* `After=network-online.target`,
otherwise systemd may correctly start your service "after" a target
that wasn't requested and therefore was never started.

### `[Service]` — how to actually run the process

```ini
[Service]
Type=simple                                    # see the table below
ExecStart=/usr/local/bin/myapp --port 8080     # MUST be an absolute path
ExecReload=/bin/kill -HUP $MAINPID             # optional; enables `systemctl reload`
ExecStop=/usr/local/bin/myapp --shutdown       # optional; default is SIGTERM
Restart=on-failure                             # no | on-failure | on-abnormal | always
RestartSec=5                                   # delay before restart (avoids crash loops)
User=myapp                                     # never run network services as root
Group=myapp
WorkingDirectory=/var/lib/myapp
Environment=LOG_LEVEL=info
EnvironmentFile=/etc/myapp/myapp.env           # one KEY=VALUE per line
# resource limits via cgroups
MemoryMax=512M
CPUQuota=50%
TasksMax=4096
# hardening (cheap, high value)
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
```

The `Type=` knob is worth spelling out:

| Type | When systemd considers the service "started" | Use it for |
|---|---|---|
| `simple` (default) | Immediately after `ExecStart=` runs. | Foreground processes — most modern daemons, scripts, anything you'd run in a container. |
| `exec` | After the binary has been `exec()`'d but before any code runs. | Like `simple`, slightly stricter readiness. |
| `forking` | After the parent process exits and a child remains. | Old-school daemons that double-fork. |
| `oneshot` | When `ExecStart=` returns 0. The unit can be `active` without any process running. | Setup tasks, mount helpers, idempotent scripts. Often paired with `RemainAfterExit=yes`. |
| `notify` | When the service calls `sd_notify(READY=1)`. | Anything that wants accurate readiness — DBs, schedulers, anything else that depends on it. |

### `[Install]` — when to enable it

```ini
[Install]
WantedBy=multi-user.target
# Alias=myapp-alt.service   (optional second name)
```

`[Install]` is read *only* by `systemctl enable` / `disable`. It tells
systemd which target should pull this unit in when it's enabled — for
99% of server services that's `multi-user.target`. Without an
`[Install]` section the unit is "static" and `enable` will refuse to
do anything.

## Editing units the right way

Distribution packages ship unit files under `/usr/lib/systemd/system/`.
**Don't edit those.** Instead, create a drop-in:

```bash
sudo systemctl edit nginx.service
```

This opens an empty file at
`/etc/systemd/system/nginx.service.d/override.conf`. Anything you put
there is merged on top of the vendor unit. To completely replace the
vendor unit (rare), use `systemctl edit --full`. After any edit:

```bash
sudo systemctl daemon-reload          # pick up the file change
sudo systemctl restart nginx          # apply it to the running process
```

`daemon-reload` only re-reads unit files; it does not restart anything.
Forgetting it is the single most common reason "my edit had no effect."

# 5. journalctl: the logs are already there

systemd ships with **journald**, a logging daemon that captures
stdout/stderr from every service plus everything written to syslog.
Nothing extra to configure — your service prints to stdout, the
journal records it.

![journalctl filters: one journal, many filters](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/service-management/fig4_journalctl_filters.png)

## The filters you need

```bash
journalctl -u nginx                        # logs of one unit
journalctl -u nginx -f                     # follow new entries (like tail -f)
journalctl -u nginx -n 100                 # last 100 lines
journalctl -u nginx --since "10 min ago"   # human time expressions work
journalctl -u nginx --since "2025-02-01" --until "2025-02-02"
journalctl -u nginx -p err                 # priority err and worse
journalctl -u nginx -o json-pretty         # full structured record
journalctl -b                              # this boot only
journalctl -b -1                           # the previous boot
journalctl -k                              # kernel messages only (dmesg-equivalent)
journalctl _PID=1234                       # records from a specific PID
journalctl _UID=1000 _COMM=python3         # combine arbitrary fields
```

Two of these are load-bearing in incident response. `journalctl -u
<svc> -p err -b` ("errors from this unit, this boot") is usually the
first command I run when investigating a failure. `journalctl -b -1`
is how you find out what happened just before a reboot you didn't
expect — those messages would otherwise be gone.

The priority levels are the syslog ones, numbered 0 (most severe) to
7 (debug): `emerg`, `alert`, `crit`, `err`, `warning`, `notice`,
`info`, `debug`. `-p err` means "level `err` and worse" — i.e.
levels 0 through 3.

## Persisting the journal across reboots

By default many distributions store the journal in `/run/log/journal/`
— in a `tmpfs`, which is wiped on reboot. That is fine until you
need the logs from before the crash. Make it persistent:

```bash
sudo mkdir -p /var/log/journal
sudo systemd-tmpfiles --create --prefix /var/log/journal
sudo systemctl restart systemd-journald
```

You can also cap the disk it uses by editing
`/etc/systemd/journald.conf` (`SystemMaxUse=2G`, `MaxRetentionSec=30day`,
etc.) and reloading `systemd-journald`.

## Cleaning up

```bash
sudo journalctl --disk-usage                 # how much space is the journal eating
sudo journalctl --vacuum-time=7d             # keep last 7 days
sudo journalctl --vacuum-size=1G             # cap at 1 GiB
```

# 6. Timers: the modern cron

`cron` still works on every Linux system. But on a systemd box,
`.timer` units are usually a better choice — they share the journal
with everything else, run as proper units (so they get restart
policies, resource limits, dependencies) and they survive missed
runs (`Persistent=true`).

A timer is always a pair: a `.timer` that fires on a schedule, and a
`.service` that does the work.

`/etc/systemd/system/backup.service`:

```ini
[Unit]
Description=Nightly database backup

[Service]
Type=oneshot
ExecStart=/usr/local/bin/backup.sh
User=backup
```

`/etc/systemd/system/backup.timer`:

```ini
[Unit]
Description=Run database backup nightly

[Timer]
OnCalendar=*-*-* 02:30:00       # every day at 02:30 local time
RandomizedDelaySec=10min        # smear across a window so a fleet doesn't stampede
Persistent=true                 # run on next boot if we missed the slot (laptop / VM)
Unit=backup.service             # which unit to trigger (defaults to same name)

[Install]
WantedBy=timers.target
```

Enable the *timer*, not the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now backup.timer
systemctl list-timers --all     # when does each timer next fire?
```

`OnCalendar=` syntax is rich — `Mon..Fri 09:00`, `hourly`, `weekly`,
`*-*-1 04:00:00` for the first of every month, and so on.
`systemd-analyze calendar 'Mon..Fri 09:00'` will tell you exactly
when an expression resolves to.

Compared to cron, the wins are: failures show up in
`systemctl --failed` and in the journal next to the rest of the
service's logs; the job inherits all of `[Service]`'s
hardening/limit knobs; and `Persistent=true` solves the laptop
problem (the cron job that "should have run at 03:00" while the
laptop was asleep simply doesn't, ever).

# 7. The boot timeline

Knowing roughly what happens between power-on and your login prompt
makes "why did boot take 90 seconds" tractable.

![Boot timeline: firmware to multi-user.target](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/service-management/fig5_boot_timeline.png)

1. **Firmware (BIOS/UEFI)** runs POST and picks a boot device.
2. **Bootloader** (typically GRUB) loads the kernel and the
   `initramfs` into memory, hands control to the kernel.
3. **Kernel** initialises hardware, mounts the root filesystem
   read-only, then starts `/sbin/init` — which is a symlink to
   systemd. From here on, PID 1 is in charge.
4. **systemd** parses unit files and starts walking the dependency
   graph, in parallel wherever it can.
5. **Targets** activate in order: `sysinit.target` (early mounts,
   swap, udev) → `basic.target` (sockets, timers, paths armed) →
   `multi-user.target` (every enabled service is up, login prompt
   ready). On a desktop, `graphical.target` follows.

Three diagnostic commands earn their keep here:

```bash
systemd-analyze                                  # total boot time, broken into kernel / userspace
systemd-analyze blame                            # services sorted by activation time
systemd-analyze critical-chain                   # the longest dependency chain (the one that matters)
systemd-analyze critical-chain nginx.service     # the chain leading to one specific unit
```

`blame` is misleading on its own — a service can take 5 seconds to
start without delaying boot at all, if nothing was waiting on it.
`critical-chain` is what tells you which slow service is actually on
the critical path. Optimise that one first.

# 8. Common services in 60 seconds

This section is a reference card for the four services you will touch
most often.

## Time synchronisation

Time skew breaks everything sooner or later — TLS handshakes, log
correlation, Kerberos, distributed-database replication. On a
systemd box the simplest answer is the built-in client:

```bash
sudo timedatectl set-ntp true
sudo timedatectl set-timezone Asia/Shanghai
timedatectl                                # current state, including "System clock synchronized: yes"
timedatectl list-timezones | grep Shanghai
```

`timedatectl` enables `systemd-timesyncd`, which is a small SNTP
client adequate for clients and most servers. If you need a real NTP
implementation (high-accuracy clusters, serving time to other hosts),
install **chrony** and disable timesyncd.

Avoid `ntpdate`. It steps the clock instantly, which can break
anything that assumed time only moves forwards (cron, journald
ordering, replication). Real NTP clients slew the clock smoothly.

## Firewall (firewalld)

```bash
sudo systemctl enable --now firewalld
sudo firewall-cmd --get-default-zone
sudo firewall-cmd --list-all                       # what's open right now

# add and persist a rule
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload                         # without --reload, --permanent rules don't apply
```

Without `--permanent` the rule lasts until the next reload; with
`--permanent` but without `--reload` it lasts until the next reload
but doesn't apply now. In practice you use both.

Zones (`public`, `work`, `home`, `internal`, `dmz`, `trusted`,
`drop`, `block`) are firewalld's way of saying "depending on which
network this interface is on, apply this ruleset." On a single-homed
server you usually leave everything in `public` and just open ports
there.

## SSH hardening

The defaults in `/etc/ssh/sshd_config` are conservative on Debian and
Ubuntu, less so elsewhere. The high-value changes:

```
Port 22                          # change only if you understand the operational cost
PermitRootLogin no               # log in as a user, then sudo
PasswordAuthentication no        # keys only — eliminates the entire brute-force category
PubkeyAuthentication yes
PermitEmptyPasswords no
MaxAuthTries 3
ClientAliveInterval 300
ClientAliveCountMax 2
```

Always validate before reloading — a bad config file can lock you out
of a remote machine:

```bash
sudo sshd -t                            # parse and exit
sudo systemctl reload sshd              # apply without dropping existing sessions
```

Add `fail2ban` if your SSH port is on the public internet and you
want IP-level rate limiting; otherwise key-only auth is already
enough to make brute force pointless.

## Cron (when timers aren't an option)

If you're on a system that isn't fully systemd, or you're maintaining
existing cron jobs:

```
# m  h  dom  mon  dow   command
30 1   *    *    *    /usr/local/bin/backup.sh
*/10 *  *    *    *    /usr/local/bin/health-check.sh
0  3   *    *    0    /usr/local/bin/weekly-cleanup.sh
```

Edit per-user crontabs with `crontab -e`, list with `crontab -l`.
Always use absolute paths inside the command — cron runs with a
near-empty `PATH`. Capture both streams (`>> /var/log/myjob.log 2>&1`)
or you'll never know why the job stopped working.

# 9. Troubleshooting playbook: the service won't start

When a service refuses to come up, work the list in order. Most of
the time you find the answer in the first two steps.

**1. Read what systemctl is telling you.**

```bash
sudo systemctl status myapp.service
```

Focus on `Active:` (the state and how long), `Main PID:` (zero means
the process is gone), the `Process:` line (what command actually ran
and what it exited with), and the last 10 log lines reproduced at the
bottom. 80% of the time the answer is right there.

**2. Get the full log for this boot.**

```bash
sudo journalctl -u myapp.service -b -p err -xe
```

`-x` adds the explanation catalog when systemd has one; `-e` jumps to
the end. If the service was restarted by `Restart=on-failure`, scroll
back through the previous attempts — they often differ.

**3. Validate the config before blaming the service.**

Most daemons ship a config-check mode:

```bash
sudo nginx -t
sudo apachectl configtest
sudo sshd -t
sudo named-checkconf
sudo postfix check
```

For a custom service, run the `ExecStart=` command by hand as the
target user:

```bash
sudo -u myapp /usr/local/bin/myapp --port 8080
```

**4. Look for port conflicts.**

```bash
sudo ss -lntp | grep 8080
sudo lsof -i :8080
```

A common failure mode: the service crashed earlier, was restarted,
and the new process can't bind because the old one is somehow still
holding the port (or because something completely different took it).

**5. Look for permission and path problems.**

```bash
ls -ld /var/lib/myapp /var/log/myapp
namei -l /var/lib/myapp/data.db        # walks every directory in the path and shows who owns it
```

If the service runs as `myapp` but its working directory is owned by
root with mode 700, it will fail to start in a way that often looks
mysterious. `namei -l` makes the search-permission chain visible.

**6. Consider the security layer.**

On RHEL-family systems SELinux can block a service even when
permissions look fine; on Ubuntu, AppArmor can do the same.

```bash
# RHEL/CentOS/Rocky
sudo ausearch -m avc -ts recent          # AVC denials in the audit log
sudo setenforce 0                        # temporarily switch to permissive (test only!)

# Ubuntu/Debian
sudo journalctl -k | grep -i apparmor
sudo aa-status
```

If switching to permissive makes the problem disappear, the fix is a
proper SELinux policy or AppArmor profile, not leaving enforcement
off.

**7. Walk the dependency graph.**

```bash
sudo systemctl list-dependencies myapp.service
sudo systemctl --failed
```

If something the unit `Requires=` is itself failed, the symptom moves
upstream. Fix that first.

# 10. Where to go next

You now have, I hope, a working model: PID 1 supervises units, units
move through a state machine, unit files describe the supervision
contract, journald records everything, and `systemctl` /
`journalctl` are the two windows you look through. From here:

- **`man systemd.service`** and **`man systemd.unit`** are the
  authoritative reference — short, dense, and worth reading once.
- **`man systemd.exec`** documents every sandboxing/limit knob
  available in `[Service]`. Most of them are free hardening.
- **freedesktop.org/wiki/Software/systemd/** — official docs and the
  excellent "systemd for Administrators" series by Lennart Poettering.
- **`systemd-analyze security <unit>`** scores each running service
  on its sandboxing posture and tells you which knobs would tighten
  it.

The next articles in this series cover **package management**
(installing the daemons you'll then turn into services) and
**process and resource management** (looking at what those services
are actually doing once they're up). The mental model from this
article — services as supervised units, with explicit dependencies
and a structured log — carries straight through.
