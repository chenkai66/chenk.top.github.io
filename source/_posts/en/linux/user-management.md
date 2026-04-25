---
title: "Linux User Management: Users, Groups, sudo, and Security"
date: 2022-02-22 09:00:00
tags:
  - Linux
  - Cloud
  - System Administration
categories: Linux
series: "Linux"
series_order: 5
series_total: 8
lang: en
mathjax: false
description: "A working mental model for Linux accounts: how /etc/passwd and /etc/shadow fit together, when to use a primary group versus a supplementary one, how sudo actually decides, and the full lifecycle of useradd / usermod / passwd / chage / userdel — including the PAM stack underneath."
disableNunjucks: true
---

If you only ever ran `useradd` and `passwd` on a single laptop, you can probably get away without thinking about any of this. The moment more than one human (or more than one service) shares a host, "user management" stops being paperwork and starts being the security model: it decides who can log in, which UID owns the files a process writes, which commands `sudo` will lift to root, and how long a stolen password remains useful.

This article walks the model end to end. We start with the raw shape of `/etc/passwd` and `/etc/shadow` — because every command in this space is just a wrapper around editing those files. Then we cover the user/group relationship (the bit people most often get backwards), the full lifecycle commands (`useradd`, `usermod`, `passwd`, `chage`, `userdel`), `sudo` and `visudo` done right, and finally the PAM stack that ties authentication, account policy, password rules and session setup together.

# The mental model: accounts are rows in three text files

Before any command, the data model. Linux accounts live in three flat text files, one row per entity, fields separated by colons:

- `/etc/passwd` — public: one row per account (humans **and** services). World-readable, root-writable.
- `/etc/shadow` — secret: one row per account holding the password hash and the aging policy. Root-only.
- `/etc/group` — one row per group, with a comma-separated member list at the end.

There is also `/etc/gshadow` (group passwords and group admins, rarely used in practice) and `/etc/skel` (the template directory copied into every new home).

![Anatomy of an /etc/passwd entry and its link to /etc/shadow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/user-management/fig1_passwd_anatomy.png)

The seven `/etc/passwd` fields, left to right:

1. **username** — the login name. Unique per host.
2. **password placeholder** — almost always `x`, meaning "the real hash is in `/etc/shadow`". A literal `*` or empty field means no usable password (different from "locked").
3. **UID** — the numeric user id. The kernel only ever sees this number; the name is a convenience for humans. `0` is root, `1–999` are reserved for services, `1000+` are humans on modern distros (CentOS 6 and earlier started humans at `500`).
4. **GID** — the *primary* group id. New files this user creates are owned by this group by default.
5. **GECOS** — historically the General Electric Comprehensive Operating System full-name field. Today: free-form comment. `chfn` edits it.
6. **home directory** — `$HOME` after login. Created from `/etc/skel` if you pass `-m` to `useradd`.
7. **login shell** — what `exec`s after authentication. `/sbin/nologin` or `/usr/sbin/nologin` makes the account non-interactive (the right setting for service accounts).

The corresponding `/etc/shadow` line carries: the hash (prefix tells you the algorithm — `$6$` is SHA‑512, `$y$` is yescrypt on newer Debian/Ubuntu, `$1$` is MD5 and you should never see it on a modern host), the day-count of the last password change (days since 1970-01-01), and five aging fields: `min`, `max`, `warn`, `inactive`, `expire`. A leading `!` or `*` on the hash means "locked" — the account exists but no password will ever match.

> Never edit these files in a normal editor. Use `vipw` and `vigr`, which take the right locks (`/etc/passwd.lock`, `/etc/shadow.lock`) so you can't corrupt the file by saving while `useradd` is also running. Better yet, use the wrapper commands.

# Users, primary group, supplementary groups

This is the part most tutorials get muddled. Every account has exactly **one primary group** (the GID in `/etc/passwd`) and **zero or more supplementary groups** (rows of `/etc/group` whose member list contains this user).

![Users belong to one primary group and many supplementary groups](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/user-management/fig2_user_group_relationship.png)

Why two kinds?

- The **primary group** decides default ownership: when `alice` runs `touch foo`, the file is owned `alice:alice` (or whatever her primary group is). It also decides what the kernel sets as the process's `egid` at login.
- **Supplementary groups** grant *extra* access. Adding `alice` to `docker` lets her talk to the docker socket; adding her to `sudo` (Debian) or `wheel` (RHEL) lets her escalate. None of this changes the default ownership of files she creates.

A common confusion: if `alice`'s primary group is `alice`, you will **not** see her name in the `/etc/group` row for `alice` — primary membership lives in `/etc/passwd`, not `/etc/group`. The member list in `/etc/group` only enumerates the *supplementary* members. To see the union, ask the kernel:

```bash
id alice
# uid=1001(alice) gid=1001(alice) groups=1001(alice),998(docker),27(sudo)
groups alice
# alice : alice docker sudo
```

# The lifecycle commands

Every account passes through the same five stages. Each command edits a specific subset of the files above; once you know which, recovery and auditing become trivial.

![The lifecycle of a Linux account: useradd, usermod, passwd, lock, userdel](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/user-management/fig4_user_lifecycle_commands.png)

## `useradd` — create

```bash
sudo useradd -m -s /bin/bash -c "Alice Wang" alice
sudo passwd alice
```

The flags worth memorising:

| Flag | Effect |
|------|--------|
| `-m` | Create the home directory (and copy `/etc/skel` into it). Without `-m` you get an account with `$HOME` set to a path that doesn't exist. |
| `-s SHELL` | Login shell. `/bin/bash`, `/bin/zsh`, or `/sbin/nologin` for service accounts. |
| `-g GROUP` | Primary group. Default is to create a same-name group with the same GID (this is "user private groups" — the standard layout on every modern distro). |
| `-G g1,g2` | Supplementary groups, comma-separated. Don't forget the `,` between them. |
| `-u UID` | Pin a UID. Useful for keeping NFS-shared filesystems consistent across hosts. |
| `-r` | Make a *system* user (UID below the normal range, no aging, no `/home`). |
| `-c "..."` | The GECOS / comment field. |

Two common shapes:

```bash
# A human, in the developers group, with docker socket access:
sudo useradd -m -s /bin/bash -G developers,docker bob
sudo passwd bob

# A service account that nginx will run under:
sudo useradd -r -s /sbin/nologin -d /var/lib/nginx -M nginx
```

`-M` says "do not create the home directory" — many service accounts don't need one because their state lives somewhere else (`/var/lib/<service>`).

## `usermod` — modify in place

The trap with `-G` is that it **replaces** the existing supplementary list. Use `-aG` to *append*:

```bash
# Right: add alice to the sudo group, keep her other groups.
sudo usermod -aG sudo alice

# Wrong: this REPLACES alice's supplementary groups with just {sudo}.
sudo usermod -G sudo alice
```

Other useful invocations:

```bash
sudo usermod -s /bin/zsh alice                  # change login shell
sudo usermod -d /data/alice -m alice            # move home dir, migrating contents
sudo usermod -l awang -d /home/awang -m alice   # rename account
sudo usermod -L alice                           # lock (prepends '!' to the hash)
sudo usermod -U alice                           # unlock
```

A locked account still exists, still owns its files, and can still be the target of `su - alice` from root. It just can't be authenticated against by password. This is exactly what you want when an employee leaves but their files might still be needed for a few weeks.

## `passwd` and `chage` — secrets and their expiry

`passwd` sets passwords; `chage` sets the policy that decides when a password must change.

```bash
passwd                      # change your own password
sudo passwd alice           # change alice's password
sudo passwd -l alice        # lock (same effect as `usermod -L`)
sudo passwd -u alice        # unlock
sudo passwd -e alice        # expire now: alice must reset on next login
sudo passwd -d alice        # delete password (passwordless login — almost never what you want)
```

`chage -l` shows the current policy; `chage` sets it:

```bash
sudo chage -l alice
# Last password change         : Apr 21, 2026
# Password expires             : Jul 20, 2026
# Password inactive            : Aug 19, 2026
# Account expires              : never
# Minimum number of days...    : 1
# Maximum number of days...    : 90
# Number of days of warning... : 7

# A reasonable policy: rotate every 90 days, warn 7 days ahead,
# disable the account if 30 days pass after the password expires.
sudo chage -m 1 -M 90 -W 7 -I 30 alice

# Hard expiry on a specific date (e.g. a contractor):
sudo chage -E 2026-12-31 alice
```

Note that "max age" alone is not the security control most people think it is — modern guidance (NIST SP 800-63B) actually argues against forced periodic rotation for *human* passwords because it pushes people to predictable patterns. Use it for *service* accounts and for compliance regimes that demand it; rely on MFA, length and breach detection for humans.

## `userdel` — and why you should lock first

```bash
sudo userdel alice          # remove the account, leave $HOME alone
sudo userdel -r alice       # remove the account AND $HOME AND mail spool
```

The hazard: any file `alice` owned that lived outside `$HOME` becomes an orphan owned by a bare UID. The next account that gets that UID **silently inherits those files**. The fix is procedural, not technical:

1. `usermod -L alice` (or `passwd -l alice`) — lock immediately, no more logins.
2. Wait. Cron jobs need to drain, open shells need to close, file ownership audits need to run.
3. `find / -uid $(id -u alice) -print` to see what she owned outside `$HOME`. Reassign with `chown` or archive.
4. Only then `userdel -r alice`.

# Group management

```bash
sudo groupadd developers              # create
sudo groupadd -g 2000 developers      # create with a fixed GID
sudo groupmod -n devs developers      # rename
sudo groupmod -g 3000 developers      # change GID (does NOT chown existing files!)
sudo groupdel developers              # delete (refuses if it's anyone's primary group)
sudo gpasswd -a alice developers      # add alice to developers
sudo gpasswd -d alice developers      # remove alice from developers
sudo gpasswd -A alice developers      # make alice a group administrator
```

`gpasswd` is the dedicated wrapper for `/etc/group` and `/etc/gshadow`; it is generally safer than poking those files via `usermod -G`, and it lets you delegate group membership management to a non-root user via the "group administrator" role.

# `sudo`: how a single command becomes root

Logging in directly as root is wrong for two reasons. First, `rm -rf /` deserves friction. Second, the audit log just shows "root did things" — completely useless when more than one person has the password. `sudo` fixes both: each call is logged with the *real* user, and the policy file lets you grant exactly the privilege needed and no more.

![How sudo decides whether to run your command](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/user-management/fig3_sudo_policy_hierarchy.png)

A `sudoers` rule has to match on **five** dimensions before the command runs:

```
user_or_%group  host=(runas_user:runas_group)  TAG=  command_list
```

Read out loud: "*who*, on *which host*, may run as *whom*, with *what tags*, *which commands*."

## Edit with `visudo`, never your editor

```bash
sudo visudo                       # /etc/sudoers
sudo visudo -f /etc/sudoers.d/ops # a drop-in file
```

`visudo` takes the lock, runs `sudoers` syntax validation on save, and refuses to install a broken file. Editing `/etc/sudoers` directly with `vim` is one of the few mistakes that can lock you out of a server with no way back short of single-user mode — because if `sudo` itself can't parse its config, *nothing* will let you fix it.

Prefer drop-ins under `/etc/sudoers.d/` (they're loaded by an `@includedir` line in the main file). They version, package, and review more cleanly than a single monolithic file.

## The shapes you actually need

```sudoers
# Full root, password required.
alice   ALL=(ALL:ALL) ALL

# Group-based: members of `wheel` (RHEL) or `sudo` (Debian) get root.
%wheel  ALL=(ALL:ALL) ALL

# Narrow: bob can ONLY restart nginx, no password prompt.
bob     ALL=(root) NOPASSWD: /usr/bin/systemctl restart nginx, \
                              /usr/bin/systemctl status nginx

# Aliases keep large policies readable.
Cmnd_Alias NGINX_CTL = /usr/bin/systemctl restart nginx, \
                        /usr/bin/systemctl reload nginx, \
                        /usr/bin/systemctl status nginx
User_Alias  ONCALL    = alice, bob, carol
ONCALL    ALL=(root) NOPASSWD: NGINX_CTL
```

A few non-obvious rules:

- The command list must be **absolute paths**. `bob ... NOPASSWD: nginx` does nothing useful and may even open a hole, because users could arrange `nginx` to mean something else on `$PATH`.
- Commands are matched as prefixes unless you pin arguments. `/usr/bin/systemctl restart nginx` allows exactly that; `/usr/bin/systemctl` allows anything systemctl can do (including `poweroff`).
- `NOPASSWD:` is convenient and dangerous. Reserve it for non-interactive automation; for humans, ask for the password.
- `Defaults requiretty` (sometimes shipped on RHEL) breaks `sudo` over non-tty channels. Disable it for automation accounts with a per-user `Defaults:bot !requiretty`.

## What `sudo` reads on disk

In order: `/etc/sudoers`, then everything under `/etc/sudoers.d/` in lexical order, then group memberships are resolved, then `Defaults` are applied. Use `sudo -ll` to dump the full effective policy for the calling user — far more reliable than re-reading the files by hand.

## `su` versus `sudo`

`su -` opens a shell as another user (default: root) given **that user's** password. `sudo` runs a command as another user given **your** password. Almost always prefer `sudo`: the audit trail is better, and you never have to share root's password.

# PAM: the layer underneath everything

`sudo`, `sshd`, `login`, `gdm`, `cron`, `su`, `passwd`, `crond` — none of them implement password checking themselves. They all delegate to PAM (Pluggable Authentication Modules), a stack of `.so` libraries configured per service in `/etc/pam.d/`. Understanding PAM is what turns "I don't know why this account can't log in" into a five-second debugging exercise.

![How PAM evaluates a login: auth, account, password, session](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/user-management/fig5_pam_auth_flow.png)

A PAM service file has up to four stacks:

- **auth** — prove identity. `pam_unix.so` checks `/etc/shadow`; `pam_sss.so` talks to SSSD/LDAP/AD; `pam_google_authenticator.so` adds TOTP.
- **account** — even if the password is right, is this account allowed to log in *now*? Aging policy, `nologin` flag, time-of-day, source host.
- **password** — only consulted on `passwd`. Strength rules (`pam_pwquality.so`), reuse rules (`pam_pwhistory.so`), then write the new hash (`pam_unix.so`).
- **session** — set up the working environment after a successful login: rlimits (`pam_limits.so`), systemd user slice (`pam_systemd.so`), create the home dir on first login (`pam_mkhomedir.so`), record `lastlog`.

Each line carries a **control flag** that decides how its result combines with the rest of the stack:

- `required` — must succeed; if it fails the whole stack fails, but PAM keeps running the rest so the user can't tell *which* line failed (that's intentional — telling them leaks information).
- `requisite` — must succeed; on failure the stack aborts immediately.
- `sufficient` — if it succeeds and no earlier `required` failed, the stack passes right away.
- `optional` — result is ignored unless it's the only module in the stack.

A practical example — turning on a strong password policy on Debian. Edit `/etc/pam.d/common-password`:

```pam
password requisite pam_pwquality.so retry=3 minlen=12 \
                   dcredit=-1 ucredit=-1 ocredit=-1 lcredit=-1 \
                   difok=4 enforce_for_root
password required  pam_pwhistory.so remember=5 use_authtok
password [success=1 default=ignore] pam_unix.so obscure use_authtok yescrypt
```

What this says: at least 12 characters, requiring at least one digit / upper / symbol / lower (`-1` means "credit at most one of these"); reject anything that shares 4+ characters with the old password; remember the last 5 hashes and refuse reuse; finally write with the modern `yescrypt` hash. Test with `passwd <yourself>` *as a non-root account* before logging out.

The single most useful debugging tool here is the journal:

```bash
sudo journalctl -u sshd -e            # what sshd saw
sudo journalctl _COMM=sudo --since "1h ago"
sudo grep "alice" /var/log/auth.log   # Debian
sudo grep "alice" /var/log/secure     # RHEL
```

If a login fails with the password definitely correct, it's almost always one of: account locked (`!` in `/etc/shadow`), shell set to `/sbin/nologin`, `pam_nologin` blocking everyone because `/etc/nologin` exists, `AllowUsers`/`DenyUsers` in `sshd_config`, or password aged past `inactive`.

# Patterns from the field

## A shared project directory

Goal: `/srv/project` is read/write for everyone in `developers`, invisible to anyone else, and any new file inside it stays group-owned by `developers` so coworkers don't accidentally lock each other out.

```bash
sudo groupadd -r developers
sudo usermod -aG developers alice
sudo usermod -aG developers bob

sudo mkdir -p /srv/project
sudo chown root:developers /srv/project
sudo chmod 2770 /srv/project   # 2 = SGID; 770 = rwx for owner+group, nothing for others
```

The `2` in `2770` is the **SGID bit on a directory**: new files inherit the directory's group instead of the creator's primary group. Without it, when `alice` (primary group `alice`) creates a file, it lands as `alice:alice`, and `bob` can't edit it.

For finer control — say, "developers can write but `carol` is read-only" — reach for POSIX ACLs:

```bash
sudo setfacl -m g:developers:rwx /srv/project
sudo setfacl -m u:carol:r-x /srv/project
sudo setfacl -d -m g:developers:rwx /srv/project   # default ACL: applies to new files
getfacl /srv/project
```

## A service account, done right

```bash
sudo useradd --system \
             --home-dir /var/lib/myapp \
             --shell /usr/sbin/nologin \
             --no-create-home \
             --user-group \
             myapp
sudo install -d -o myapp -g myapp -m 0750 /var/lib/myapp /var/log/myapp
```

`--system` gets a UID below the human range and skips aging. `--user-group` makes the same-name group. `--no-create-home` because the state directory is created explicitly, with the right mode. The systemd unit then runs as `User=myapp Group=myapp`, which is the *only* identity that should ever own the app's data on disk.

## Tiered `sudo`

```sudoers
Cmnd_Alias READ_LOGS = /usr/bin/journalctl, /usr/bin/tail, /usr/bin/less
Cmnd_Alias NGINX_CTL = /usr/bin/systemctl restart nginx, \
                        /usr/bin/systemctl reload nginx, \
                        /usr/bin/systemctl status nginx

# Full admin
alice    ALL=(ALL:ALL) ALL
# On-call: only nginx, no password (paged at 3am, can't fumble auth)
bob      ALL=(root) NOPASSWD: NGINX_CTL
# Support: read logs, with password
carol    ALL=(root)          READ_LOGS
```

## Bulk creation from a CSV

```bash
#!/usr/bin/env bash
set -euo pipefail
while IFS=, read -r username fullname groups; do
  id "$username" &>/dev/null && { echo "skip $username (exists)"; continue; }
  sudo useradd -m -s /bin/bash -c "$fullname" -G "$groups" "$username"
  # Generate a one-time random password and force change on first login.
  pw=$(openssl rand -base64 16)
  echo "$username:$pw" | sudo chpasswd
  sudo chage -d 0 "$username"     # 'last change = epoch' -> must reset on first login
  echo "$username,$pw" >> /root/initial-passwords.csv
done < users.csv
```

`chpasswd` reads `user:password` lines and is the right tool for batch updates; `passwd --stdin` is RHEL-only. `chage -d 0` is the trick that forces a reset on first login without giving the user a non-expiring random secret.

# Where to go next

This article gave you the operational model: the file shapes, the lifecycle commands, the `sudo` policy language and the PAM stack. The next two pieces in the series build on it directly:

- [Linux File Permissions](../file-permissions/) — `rwx`, the SUID/SGID bits we used for the shared directory above, and POSIX ACLs.
- [Linux System Service Management](../service-management/) — how the `User=` / `Group=` / `DynamicUser=` directives in systemd units make service accounts mostly redundant.

Worth reading on the side: `man 5 sudoers`, `man 8 pam.d`, `man 5 shadow`, and the FreeIPA / SSSD documentation if you ever need centralised identity for more than a handful of hosts.
