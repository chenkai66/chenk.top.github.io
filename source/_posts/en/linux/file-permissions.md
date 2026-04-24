---
title: "Linux File Permissions: rwx, chmod, chown, and Beyond"
date: 2024-01-20 09:00:00
tags:
  - Linux
  - Cloud
  - System Administration
categories: Linux
series: "Linux"
series_order: 2
series_total: 8
lang: en
mathjax: false
description: "Master the Linux permission model: rwx semantics on files vs directories, numeric and symbolic notation, chmod/chown usage, umask defaults, SUID/SGID/Sticky bit, and ACLs."
disableNunjucks: true
---

File permissions look elementary — `chmod 755`, done — but they remain one of the top causes of production incidents I see: a service won't start, a deploy script silently does nothing, Nginx returns `403`, a shared directory leaks, or `rm` refuses on a file that "should" be removable. Memorising magic numbers does not get you out of any of these. What does is understanding three things at the same time:

1. The **same `r`/`w`/`x` bits mean different things on a regular file than on a directory** — the directory case is what trips most people up.
2. The kernel's check uses **owner / group / others as a 3-step `if/else if/else`**, not as a sum — so being in the group is sometimes worse than being "everyone else."
3. `umask`, `setuid`, `setgid`, sticky bit, and ACLs all exist for **a specific reason**, and using them outside that reason is how systems get owned.

This article works through the model from the smallest concept up: bit semantics, numeric vs symbolic notation, `chmod`/`chown`/`chgrp`, default permissions via `umask`, the three special bits, ACL with `getfacl`/`setfacl`, and a concrete troubleshooting checklist. Examples are the kind you actually meet — webroot, shared team folders, `/tmp`, immutable config files — not toy puzzles.

# The permission model: owner / group / others

![Mode string anatomy and rwx on files vs directories](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/file-permissions/fig1_rwx_bits.png)

Linux is a multi-user system, so every inode (file, directory, symlink, device, …) carries three identity hooks:

- **Owner (`u`)**: the UID that owns the inode. Usually whoever created it.
- **Group (`g`)**: a single GID. Members of that group share group-level access.
- **Others (`o`)**: every authenticated principal that is *neither* the owner *nor* in the group.

The kernel's access check is not "add up the matching bits." It is a strict **first-match cascade**:

```
if  caller.uid == file.uid          -> use OWNER bits, decision is final
elif caller.gid_set ∩ {file.gid}    -> use GROUP bits, decision is final
else                                 -> use OTHERS bits
```

That ordering has a practical consequence that surprises people: **if the owner's bits forbid an action, being in the group does not help**. `chmod 047 myfile` makes the owner unable to read their own file even though "others" can read/write/execute it.

`root` (uid 0) bypasses the check entirely via `CAP_DAC_OVERRIDE`. The sticky bit is the one exception in the other direction — see below.

## The 10-character mode string

`ls -l` prints something like `-rwxr-xr-x`. Read it left to right:

| Position | Meaning |
|----------|---------|
| 1 | File type: `-` regular, `d` directory, `l` symlink, `c`/`b` char/block device, `s` socket, `p` FIFO |
| 2–4 | Owner bits `rwx` |
| 5–7 | Group bits `rwx` |
| 8–10 | Others bits `rwx` |

Each bit is one of:

- `r` (4) — read
- `w` (2) — write
- `x` (1) — execute (or, on a directory, "traverse")

Sum within a triplet to get the octal digit; concatenate the three digits to get the familiar `755`, `644`, `600`. Where you see `s`/`S`/`t`/`T` in the `x` slot, a special bit is also set — keep reading.

# rwx on files vs directories — the most common pitfall

This is where most permission bugs live. Same letters, different semantics.

## On a regular file

| Bit | Means | Without it you can't |
|-----|-------|----------------------|
| `r` | read the bytes | `cat`, `less`, `cp src=...` |
| `w` | overwrite, truncate, or `O_TRUNC` open | `>`, in-place edit |
| `x` | exec the file as a program (must have a valid header — ELF, or a shebang for scripts) | `./prog` |

Note that `w` is **only** about modifying the file's *contents*. You do **not** need write to delete the file — that is controlled by `w` on the *parent directory*.

## On a directory

| Bit | Means | Without it you can't |
|-----|-------|----------------------|
| `r` | list the names of entries | `ls dir/` |
| `w` | create, delete, or rename entries (requires `x` too) | `touch dir/x`, `rm dir/x`, `mv` within dir |
| `x` | look up a name in the directory and traverse through it | `cd dir`, `cat dir/known-name`, opening any path that contains `dir` as a component |

Three quick experiments make the rules concrete:

**Case A — `r` without `x` (mode 644)**
```bash
chmod 644 mydir
ls mydir              # OK: lists names
cd mydir              # DENIED: no traverse bit
cat mydir/file.txt    # DENIED: pathname lookup needs x on every component
```

**Case B — `x` without `r` (mode 311)**
```bash
chmod 311 mydir
ls mydir              # DENIED: cannot enumerate
cd mydir              # OK
cat mydir/file.txt    # OK *if you happen to know the filename*
```
This is the basis of "private bin" tricks — make the dir traversable but not listable.

**Case C — `w` without `x` (mode 622)**
```bash
chmod 622 mydir
touch mydir/newfile   # DENIED: cannot even reach the directory
```
`w` on a directory is useless without `x`. Always pair them.

**Rule of thumb for directories: `x` is the load-bearing bit. `w` only matters together with `x`. `r` is convenience.**

# chmod: numeric vs symbolic notation

![Numeric vs symbolic notation reaching the same target](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/file-permissions/fig2_chmod_notations.png)

Both notations end up writing the same nine bits. They differ in whether you are stating an **absolute** target or making a **relative** edit.

## Numeric notation — absolute

Sum the bits per identity (`r=4`, `w=2`, `x=1`), concatenate three digits:

```bash
chmod 755 script.sh   # rwxr-xr-x  — typical executable
chmod 644 file.txt    # rw-r--r--  — typical data file
chmod 600 secret.key  # rw-------  — private key, ssh key, .env
chmod 700 ~/.ssh      # rwx------  — private dir
chmod 777 shared      # rwxrwxrwx  — almost always wrong
```

Use it when you want a **known final state**: scripted deploys, fresh files where you don't care what was there before.

## Symbolic notation — relative

`who` (`u`, `g`, `o`, `a` for all) + operator (`+`, `-`, `=`) + bits (`r`, `w`, `x`, `X`, `s`, `t`):

```bash
chmod u+x  script.sh        # add x for owner only
chmod go-w file.txt         # remove w for group AND others
chmod o=r  file.txt         # set others to exactly r--
chmod a+r  notes.md         # everyone can read
chmod u=rwx,g=rx,o=    dir  # multiple clauses, comma-separated
```

The capital `X` is the killer feature for trees:

```bash
chmod -R u=rwX,go=rX  project/
```

`X` adds `x` **only to directories and to files that already have at least one `x` bit**. Without it, `chmod -R 755 project/` makes every `.md`, `.png`, and `.csv` "executable" — harmless but ugly, and a gift to anyone scanning for misconfigured webroots.

Use symbolic when you want to **tweak one dimension** without disturbing the rest.

# chown / chgrp: changing ownership

```bash
sudo chown alice file               # change owner only
sudo chown alice:devs file          # change owner AND group
sudo chown :devs file               # change group only (or use chgrp)
sudo chgrp devs file                # same as above
sudo chown -R alice:devs project/   # recurse
sudo chown --reference=template new # copy ownership from another file
```

Ground rules:

- Only **root** can change ownership freely. A regular user cannot give a file away (otherwise users would dodge disk quota by re-parenting their files to someone else).
- The owner can `chgrp` to **any group they themselves belong to**. They cannot move a file into a group they're not a member of.
- `chown` resets `setuid`/`setgid` bits on regular files for safety — important to remember if you `chown root` a SUID binary, you must re-set the SUID afterwards.

Common patterns you'll write a hundred times:

```bash
sudo chown -R www-data:www-data /var/www/html       # webroot
sudo chown -R :developers       /srv/project        # team folder
sudo chown -R postgres:postgres /var/lib/postgresql # database files
```

# umask: the default-permission filter

`umask` is the **mask of bits to subtract** from the system default when a process creates a new inode. The system default is:

- `0666` for regular files (no `x` — preventing accidentally-executable data)
- `0777` for directories (`x` is needed for traversal)

Effective permissions = default `AND NOT umask`.

| umask | new file | new dir | who is this for |
|-------|----------|---------|-----------------|
| `022` | 644      | 755     | desktop default; world-readable |
| `027` | 640      | 750     | **server / production** — group-only outside owner |
| `002` | 664      | 775     | dev shared workstations with a per-user primary group (USERGROUPS) |
| `077` | 600      | 700     | strictest — `~/.ssh`, secrets dirs |

Inspect and change:

```bash
umask              # 0022 — leading 0 is the special-bit slot, ignore it
umask 027          # current shell only
echo 'umask 027' >> ~/.bashrc   # per-user persistent
```

System-wide defaults live in `/etc/login.defs` (`UMASK`) and `/etc/profile` / `/etc/pam.d/*`. On systemd units, set `UMask=` in the unit file rather than relying on shell config — services don't read `~/.bashrc`.

# Special permission bits: SUID, SGID, sticky

![SUID, SGID, and sticky bit at a glance](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/file-permissions/fig4_special_bits.png)

`chmod` actually takes a **four-digit** octal. The leading digit packs three flags: `4` (SUID), `2` (SGID), `1` (sticky). They can combine: `chmod 6755` sets SUID + SGID + 755.

## SUID (4xxx) — run as owner

When set on an executable, the process runs with the **file owner's** effective UID, regardless of who launched it. The canonical example:

```bash
$ ls -l /usr/bin/passwd
-rwsr-xr-x 1 root root … /usr/bin/passwd
#    ^  the s in the owner-x slot means SUID
```

`passwd` needs to write `/etc/shadow` (mode `0640 root:shadow`) but unprivileged users have to be able to change their own password. SUID + a tiny, audited program is the classic answer.

```bash
chmod u+s prog        # symbolic
chmod 4755 prog       # numeric (4 = SUID)
chmod u-s prog        # remove
```

`s` (lowercase) means SUID **and** the underlying `x` is set; `S` (uppercase) means SUID is set but `x` is **not** — almost always a misconfiguration.

**SUID is genuinely dangerous.** A bug in a SUID-root binary becomes a local privilege escalation. Audit them periodically:

```bash
sudo find / -xdev -perm -4000 -type f 2>/dev/null
```

Anything outside the standard set (`passwd`, `sudo`, `mount`, `su`, `ping` on older distros, …) deserves a justification.

## SGID (2xxx) — two distinct uses

**On an executable**: the process runs with the file's group as its effective GID. Used by tools that need access to a private group resource (e.g., `wall` writes to `/dev/tty*` owned by `tty`).

**On a directory** (the much more common use): files and subdirectories created inside **inherit the directory's group** instead of the creator's primary group. This is the right way to build a team-shared folder:

```bash
sudo mkdir /srv/project
sudo chown :developers /srv/project
sudo chmod 2770 /srv/project        # SGID + rwxrwx---
```

Now anyone in `developers` who creates a file inside automatically gets `group=developers`, so other team members can read/write it. Without SGID you would have to remember to `chgrp` every file you create — and people will forget.

## Sticky bit (1xxx) — restricted delete

On a directory, the sticky bit changes one rule: **only the file's owner (or root) may unlink or rename entries**, even though the directory itself is world-writable. `/tmp` is the canonical case:

```bash
$ ls -ld /tmp
drwxrwxrwt 17 root root … /tmp
#         ^ the t in the others-x slot is the sticky bit
```

Without sticky, `/tmp` (mode `1777`) would be a free-for-all where anyone could `rm` anyone else's session sockets, lock files, etc.

```bash
chmod +t  dir         # symbolic
chmod 1777 /tmp       # numeric
```

Sticky on a *file* exists historically (used to mean "keep text segment in swap") and is ignored on modern Linux.

# Common scenarios — copy-pasteable, with reasoning

![Owner / Group / Others decision matrix on a real shared dir](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/file-permissions/fig3_ugo_matrix.png)

## 1. "Permission denied" running a script

```bash
$ ./deploy.sh
zsh: permission denied: ./deploy.sh
$ ls -l deploy.sh
-rw-r--r-- 1 alice alice 432 Jan 18 09:14 deploy.sh
```

No `x`. Fix:

```bash
chmod u+x deploy.sh        # only owner needs to run it
# or, if it lives in a shared bin/
chmod 755 deploy.sh
```

If you still see "exec format error," the script is missing a shebang (`#!/usr/bin/env bash` on the first line) and the kernel doesn't know what interpreter to use.

## 2. Web server returns 403

`nginx`/`apache` runs as `www-data` (Debian/Ubuntu) or `nginx` (RHEL family). It needs:

- `r` on the file it serves
- `x` on **every directory in the path** down to that file

The second part is what bites people — `/home/alice/site/` typically has mode `700`, so `www-data` cannot even traverse into it. Either move the docroot under `/var/www/`, or open the path:

```bash
sudo chown -R www-data:www-data /var/www/html
sudo find /var/www/html -type d -exec chmod 755 {} \;
sudo find /var/www/html -type f -exec chmod 644 {} \;
# equivalent in one shot:
sudo chmod -R u=rwX,go=rX /var/www/html
```

## 3. Team-shared project directory

Goal: everyone in `developers` can read and write everything; nobody else can even peek.

```bash
sudo mkdir /srv/project
sudo chown :developers /srv/project
sudo chmod 2770 /srv/project        # SGID + rwxrwx---
# also nudge umask so new files are group-writable
echo 'umask 002' | sudo tee /etc/profile.d/team-umask.sh
```

`2` (SGID) makes inheritance work; `770` keeps outsiders out; `umask 002` ensures new files end up `664` rather than the default `644`, so other team members can edit them.

## 4. Multi-user temp directory

Already done by your distro — `/tmp` is `1777`. If you need a similar shared scratch space:

```bash
sudo mkdir /srv/scratch
sudo chmod 1777 /srv/scratch
```

## 5. Locking down a private key

```bash
chmod 600 ~/.ssh/id_ed25519
chmod 700 ~/.ssh
```

`ssh` will outright refuse to use a key that is group- or world-readable. This is a feature.

# ACL: when three buckets aren't enough

![Classic UGO vs ACL extension](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/file-permissions/fig5_acl_extension.png)

Classic mode bits give you exactly three buckets. Real-world requirements often need more: "let auditor `eve` read this report, but she's not in `developers`," or "block one specific contractor from this folder." That is what POSIX ACLs are for.

## Reading ACLs

```bash
$ getfacl report.csv
# file: report.csv
# owner: alice
# group: developers
user::rw-
user:eve:r--           # extra entry — eve can read
group::r--
group:qa:rw-           # extra entry — qa can read+write
mask::rw-              # ceiling for "extra" entries (see below)
other::---
```

A trailing `+` in `ls -l` (e.g. `-rw-r-----+`) is the visible signal that ACL entries are present.

## Setting ACLs

```bash
setfacl -m u:eve:r       report.csv     # grant eve read
setfacl -m g:qa:rw       report.csv     # grant qa group read+write
setfacl -m u:mallory:--- report.csv     # explicitly deny mallory
setfacl -x u:eve         report.csv     # remove eve's entry
setfacl -b               report.csv     # strip all extras, back to plain UGO
```

For directories, `-R` recurses, and **default ACLs** propagate to new children — much like SGID, but per-user:

```bash
setfacl -R -m   u:alice:rwx,g:devs:rwx project/   # apply to everything that's there
setfacl -R -d -m u:alice:rwx,g:devs:rwx project/  # also apply to anything created later
```

## The ACL mask

The `mask::` line is the maximum effective permission for any entry **except** `user::` (the owner) and `other::`. `chmod g=...` on an ACL'd file edits the mask, not the actual group entry — which is a frequent surprise. To set the group entry directly:

```bash
setfacl -m g::r-- report.csv    # set the group entry, leave mask alone
setfacl -m m::rw  report.csv    # set the mask
```

Filesystem must be mounted with ACL support. On modern ext4/xfs/btrfs this is the default; check with `tune2fs -l` or `mount | grep acl`.

# chattr / lsattr: filesystem-level attributes

`chattr` writes ext4/xfs attributes that sit **below** the permission system — they apply even to root.

```bash
sudo chattr +i /etc/resolv.conf      # immutable: nobody can edit, delete, or rename
sudo lsattr /etc/resolv.conf         # ----i---------------- /etc/resolv.conf
sudo chattr -i /etc/resolv.conf      # remove before editing

sudo chattr +a /var/log/audit.log    # append-only: cannot truncate or overwrite
echo entry >> /var/log/audit.log     # OK
echo entry >  /var/log/audit.log     # OPERATION NOT PERMITTED
```

Use `+i` to pin critical config (`/etc/fstab`, `/etc/passwd`, `/etc/sudoers`) so a fat-fingered `sed -i` can't destroy a recovery boot. Use `+a` on log files to make tamper-after-the-fact harder. Both are the right answer to "how do I stop root deleting this by accident" — short of `chattr -i` first, root can't.

# Troubleshooting checklist

A short, ordered sequence covers ~90% of permission bugs.

## Step 1 — who am I, really?

```bash
whoami                              # interactive shell
id                                  # also lists every group I'm in
ps -o user,uid,gid,cmd -p $(pidof nginx)   # for a service, check the actual runtime user
```

If a service is the consumer, the relevant identity is the systemd `User=` / `Group=`, not your login.

## Step 2 — every directory in the path

`r`+`x` on the leaf file isn't enough; the kernel re-checks `x` on every component. The fastest tool is `namei`:

```bash
$ namei -l /var/www/html/index.html
f: /var/www/html/index.html
drwxr-xr-x root     root     /
drwxr-xr-x root     root     var
drwxr-xr-x root     root     www
drwxr-x--- alice    alice    html        # <-- 'others' has no x; www-data is 'others' here
-rw-r--r-- alice    alice    index.html
```

The first line where the relevant identity lacks `x` is the culprit.

## Step 3 — ACL? attribute? mount option?

```bash
getfacl  file        # is there a + ACL hiding the rule?
lsattr   file        # is +i or +a pinning it?
mount | grep ' on / '   # is the fs mounted ro? noexec? nosuid?
```

`/tmp` mounted `noexec` will silently refuse `./script.sh` no matter what `chmod` you ran.

## Step 4 — selinux / apparmor

On RHEL/Fedora/CentOS, `getenforce` says `Enforcing`? Then `ls -lZ file` shows the SELinux label and `audit2why` explains the latest denial. On Ubuntu, `aa-status` lists AppArmor profiles. These can deny access even when classic permissions say "allowed."

## Specific symptoms

**`Permission denied` on a known-good script** → missing `x`, missing shebang, or `noexec` mount.

**Web server `403`** → `www-data` is "others" everywhere; check `namei -l` for a missing `x` along the path.

**`rm: cannot remove ...: Operation not permitted`** (note: *not* "Permission denied") → `lsattr` will show `+i`. `chattr -i` first.

**You can write to a file you don't own** → check the *parent directory's* `w` bit. Owning the file is irrelevant for delete/rename.

# Mental model and further reading

Three patterns will carry you through almost every real situation:

1. **Files vs directories**: on a file, `x` is "is it a program." On a directory, `x` is the *only* gate to even reaching the contents.
2. **First-match cascade**: owner OR group OR others — never additive. Audit by asking "which bucket does this caller fall into?"
3. **Special bits exist for one job each**: SUID = "let unprivileged callers do this exact privileged thing"; SGID-on-dir = "team folder"; sticky = "shared writable space without mutual sabotage." Outside those jobs, don't set them.

Where to go next:

- `man 1 chmod`, `man 2 chmod`, `man 5 acl`, `man 1 chattr` — the authoritative references, surprisingly readable.
- **Linux User Management** (next article in this series) — `/etc/passwd`, `/etc/shadow`, groups, sudoers, PAM.
- **Linux Pipelines and Redirection** — building on file descriptors and `stdin`/`stdout`/`stderr`.
- **MAC frameworks**: SELinux (RHEL family) and AppArmor (Debian/SUSE family) layer mandatory access control on top of the discretionary model covered here. Same questions, different answers.

If you can now read `drwxr-s---+ 4 alice developers` and immediately tell me the owner, the group, the special bit that's set, the fact that an ACL is in play, and what `bob in developers` versus `eve` outside it can each do — you have the model. The rest is muscle memory.
