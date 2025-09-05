<!-- Git Adventure Guide -->

**Author**: Andrew Rosemberg

**Date**: 2025-08-06

# 🚀 Git Adventure Guide
*A quirky journey from zero to hero with Git*

> Git is like a time machine for your code, a collaboration wizard, and a safety net all rolled into one. This guide will take you through the magical world of Git with fun examples, tips, and mini quests to level up your skills.

```
                               /\                /\                /\
                              /  \              /  \              /  \
                             / /\ \            / /\ \            / /\ \
                            / /  \ \          / /  \ \          / /  \ \
                           / /____\ \        / /____\ \        / /____\ \
                          /_/      \_\      /_/      \_\      /_/      \_\
  ___________________________________________________________________________
 /                                                                           \
|    ____ _ _       _     _                _____                              |
|   / ___(_) |_ ___| |__ (_)_ __   __ _   / ____|                             |
|  | |  _| | __/ __| '_ \| | '_ \ / _` | | |  __  ___ _ __                    |
|  | |_| | | || (__| | | | | | | | (_| | | | |_ |/ _ \ '__|                   |
|   \____|_|\__\___|_| |_|_|_| |_|\__, | | |__| |  __/ |                      |
|                                |___/   \_____|\___|_|                      |
|                                                                             |
|                  ~ ~ ~  G I T   A D V E N T U R E  ~ ~ ~                    |
 \___________________________________________________________________________/
                          \      /            \      /            \      /
                           \____/              \____/              \____/
```

Git is a version control system that helps you track changes in your code, collaborate with others, and manage different versions of your projects. Whether you're a solo developer or part of a team, mastering Git is essential for modern software development.

## 📜 Table of Contents
1. 🌱 [Getting Started](#-getting-started)
2. 📚 [Core Concepts](#-core-concepts)
3. 🪄 [Everyday Workflow](#-everyday-workflow)
4. 🌳 [Branching & Merging](#-branching--merging)
5. 🤝 [Collaborating on GitHub](#-collaborating-on-github)
6. 🧹 [Undo & Fix](#-undo--fix)
7. 🛠️ [Advanced Magic](#️-advanced-magic)
8. 🎮 [Mini Quests](#-mini-quests)
9. 🧾 [Cheat Sheet](#-cheat-sheet)
10. 🏁 [Next Steps](#-next-steps)

---

## 🌱 Getting Started
**Install Git**

Git is open-source and available on all major platforms. Here’s how to get it:

```bash
# macOS
brew install git

# Debian/Ubuntu
sudo apt install git

# Windows
scoop install git
```

**Set Your Identity**

Versions need an author! Configure your name and email so Git knows who you are:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

> 🧩 *Puzzle*: Run `git config --list` — how many settings can you spot?

---

## 📚 Core Concepts
| Concept | What It Means | Emoji Memory Hack |
|---------|---------------|-------------------|
| Repository (Repo) | A project folder under Git’s watch | 🗂️ |
| Commit | A snapshot of your code | 📸 |
| Branch | A timeline of commits | 🌿 |
| Merge | Combine branches | 🔀 |
| Remote | A repository living elsewhere | 🌐 |

> 🤓 **Did you know?** Commits are *immutable*. They’re like fossils—once created, they never change!

---

## 🪄 Everyday Workflow
0. **Initialize a new repository**  
If you’re starting a new project, create a new Git repository:

```bash
git init my-awesome-project
cd my-awesome-project
```

> 🧩 *Puzzle*: What happens if you run `git init` in a directory that already has a `.git` folder?

> 🏁 *Your Journey Starts*: Initialize a new repository.

1. **Clone** a repo to your local machine
You can create a repository directly on GitHub or any other Git hosting service, which will give you a URL to clone it.

Once you have a remote repository, you can clone it to your local machine:

```bash
git clone <url>
```
This process will also save the remote repository URL to a variable called `origin`, which is the default name for the remote repository. A repository remote is a version of your repository that is hosted on the internet or another network, allowing you to collaborate with others.

> 🧩 *Puzzle*: Can you have multiple remote copies' URLs stored in your local git configuration?

> 🏗️ *Quest Preparation*: Create a repository on Github. Then, since we started our repository locally, add the URL manually: 

```bash
git remote add origin <url>
```

2. **Hack away**  
Changes can be bundled or committed separately. Staging is the process of preparing files for a commit. 
You can stage specific files or all changes:

```bash
git add <files>      # stage
git commit -m "feat: add lunar lasers"  # snapshot
```

Commits (versions) will store your changes with a message describing what you did.

> 🧩 *Puzzle*: How do you stage all current changes at once?

> 🏗️ *First Quest*: Create a few files and make your first commit(s)

3. **Push** your masterpiece to the remote repository:
In order to collaborate with others, you need to push your changes to the remote repository:

```bash
git push origin main
```

> 🌟 *Pro‑tip*: Use `git add -p` to stage hunks interactively like a choose‑your‑own‑adventure book.

> 🏗️ *Second Quest*: After making a few commits, then push them to the remote repository.

---

## 🌳 Branching & Merging
A branch is a separate line of development. It allows you to work on features or fixes without affecting the main codebase. This is especially useful for experimenting or developing new features and avoiding breaking the main code besides avoiding conflicts with other developers' work.
Create and jump to a branch:

```bash
git switch -c <branch-name>
```

> 🧩 *Pro-tip*: Name your branch something that helps others infer the author and what you are developing without them needing to look at the meta information of your commits. For example: `ar/feature/gpu-compatibility`, `ar/bugfix/memory-leak`, or `ar/hotfix/security-patch`.

Merge it back - move your changes into the main branch:

```bash
git switch main
git merge feature/space-pizza
```

If others have made changes to the main branch, you might need to resolve conflicts.
The merge command you move you to a temporary state where you can resolve conflicts before finalizing the merge.
Git will tell you which files are in conflict by marking them in the output. You can also see the conflicts by running:

```bash
git status
```
Resolve conflicts like a diplomat—edit the conflicted files, then:

```bash
git add .
git commit -m "fix: resolve cosmic cheese conflict"
```

> ⚔️ **Boss Battle**: Create two conflicting branches and practice the merge dance.

---

## 🤝 Collaborating on GitHub
1. **Fork** ⇒ get your own copy of a repository to work on.
 - Click the "Fork" button on the top right of the repo page.

> 🧩 *Puzzle*: What’s the difference between forking and cloning?

> 🏗️ *Third Quest*: Fork the repository of this class.

2. **Clone** your fork.
In order to work on your forked repository, you need to clone it to your local machine:

```bash
git clone <url-of-your-fork>
cd <your-fork-directory>
```
3. **Sync** with upstream.  
If the original repository has new changes, you can keep your fork up to date by syncing it with the upstream repository:

```bash
git remote add upstream <url-of-original>
git fetch upstream
git rebase upstream/main
```  
However, this is much easier on the GitHub website.  
   - Go to your forked repository on GitHub and sync it with the original repository by clicking the "Sync fork" button.

4. **Push** & create a **Pull Request**.
Pull Requests (PRs) are how you propose changes to a project. After pushing your changes to your fork, go to the original repository and click "New Pull Request".

> 🧩 *Puzzle*: What’s the difference between a Pull Request and a Merge?

> 🏗️ *Fourth Quest*: Make a change in your fork, push it, and create a Pull Request to the original repository. Do this to add your name to your chosen *Lecture* (Both in the Readme file and in the class folder markdown file) and/or correct any of my many spelling mistakes.

5. Celebrate with GIFs. 🎉

---

> 🏆 *End*: End of your first adventure. The rest is important to know so read it for the in-class test, but I will let you practice once it makes sense in your daily workflow.

## 🧹 Undo & Fix
Git is a powerful tool, but sometimes you make mistakes. Here are some common scenarios and how to fix them:

| Situation | Command | Explanation |
|-----------|---------|-------------|
| Un‑stage a file | `git restore --staged file` | Put it back in working area |
| Amend last commit | `git commit --amend` | Edit message or add files |
| Time‑travel (soft) | `git reset --soft HEAD~1` | Keep changes staged |
| Time‑travel (hard) | `git reset --hard HEAD~1` | WARNING: destroys work |
| Deleted a branch too soon | `git reflog` | Find the commit hash & resurrect |

---

## 🛠️ Advanced Magic
- **Stash spells**  

```bash
git stash push -m "WIP dragon taming"
git stash list
git stash pop
```

- **Bisect detective**  

```bash
git bisect start
git bisect bad          # current commit is broken
git bisect good v2.0.0  # last known good tag
```
  Git walks the commit tree to find the culprit!

- **Cherry‑pick delicacy**  

```bash
git cherry-pick <hash>
```

---

## 🧾 Cheat Sheet

```bash
# add everything
git add -A

# show history graph
git log --oneline --graph --all --decorate

# rename a branch
git branch -m old-name new-name

# delete local & remote branch
git branch -d feature
git push origin --delete feature
```

---

## 🏁 Next Steps
- Watch [Oh My Git!](https://ohmygit.org) interactive game.
- Follow the legendary **Pro Git** book.
- Explore GitHub Actions for automation.

> ✉️ **Send feedback**: open an issue or PR—Git isn’t just code; it’s conversation!

---

### © 2025 Git Adventure Guide
