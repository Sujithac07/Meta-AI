# Connect this project to GitHub (from scratch)

This guide assumes you have **never** used Git or GitHub before. Read it in order.

---

## What you are doing (in one picture)

- **Git** is a program on your computer that tracks **versions** of your files (save points).
- **GitHub** is a **website** that stores a copy of your project online, so you can back it up, share it, and run **automated checks** (like the tests in `.github/workflows/ci.yml`).

You will: install Git, create a free GitHub account, create an empty repository on GitHub, then **link** your project folder to that repository and **upload** (push) your code.

---

## Part A — Install Git on Windows

1. Open: [https://git-scm.com/download/win](https://git-scm.com/download/win)
2. Run the installer. Safe defaults are fine; ensure **“Git from the command line and also from 3rd-party software”** is selected so `git` works in PowerShell.
3. Close and reopen **PowerShell** (or **Command Prompt**) after installation.
4. Check it works:

```powershell
git --version
```

You should see something like `git version 2.x.x`.

---

## Part B — Create a GitHub account and a new repository

1. Open [https://github.com](https://github.com) and **sign up** (or sign in).
2. Click the **+** menu (top right) → **New repository**.
3. Choose a **repository name** (for example `meta_ai_builder_pp`).
4. Choose **Public** or **Private** (private = only you and people you invite).
5. **Do not** add a README, `.gitignore`, or license **if** your computer already has a full project (you do). That avoids merge conflicts on the first push.
6. Click **Create repository**.

GitHub will show a page with setup hints. Keep that tab open; you will need the **repository URL** in Part D. It looks like:

```text
https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

---

## Part C — Before you upload: protect secrets

Your project may use a file named **`.env`** for API keys. That file must **never** be committed.

- Confirm **`.env`** is listed in **`.gitignore`** (it should be).
- Never run `git add .env` on purpose.

If you are unsure, ask someone to check before you push, or only add files explicitly (see Part D).

---

## Part D — Commands on your computer (PowerShell)

### 1. Go to your project folder

```powershell
cd "C:\Users\sujit\OneDrive\Desktop\meta_ai_builder_pp"
```

(Change the path if your folder is elsewhere.)

### 2. See what Git knows about your project

```powershell
git status
```

- **Branch name** (for example `master` or `main`) appears at the top.
- **Changes** listed are normal if you have been editing files.

### 3. Optional: see existing remotes

```powershell
git remote -v
```

You might see **no** remote, or a remote named **`origin`**, or (in this project) another name such as **`space`** pointing to Hugging Face. That is fine: you can **add GitHub as another remote** without deleting the old one.

### 4. Stage the files you want on GitHub

To stage **everything** that Git is allowed to track (respecting `.gitignore`):

```powershell
git add -A
```

To stage only specific paths (safer if you want control):

```powershell
git add README.md docs/
```

### 5. Save a version on your machine (commit)

```powershell
git commit -m "Describe your changes in one short sentence"
```

- If Git says **nothing to commit**, either there are no changes, or you need to `git add` first.
- If Git says **please tell me who you are**, run (use your email and name):

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

Then run `git commit` again.

### 6. Add GitHub as a remote (first time only)

Replace the URL with **your** repository URL from Part B:

```powershell
git remote add github https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

- The name **`github`** is arbitrary but clear; some tutorials use **`origin`**. If `origin` already exists, using **`github`** avoids overwriting it.

To check:

```powershell
git remote -v
```

### 7. Push your branch to GitHub

If your branch is **`master`**:

```powershell
git push -u github master
```

If your branch is **`main`**:

```powershell
git push -u github main
```

**Your computer is `master` but GitHub’s default branch is `main`:** push the local branch to the remote name `main`:

```powershell
git push -u github master:main
```

**GitHub already has a README (or any commit) and push is rejected** (“unrelated histories” or “non-fast-forward”): your local project and the GitHub repo started as different histories. After you have committed locally, if you intend **your folder to replace** what is on GitHub (typical for a new repo with only the template README), use:

```powershell
git push -u github master:main --force-with-lease
```

`--force-with-lease` overwrites the remote only if no new commits appeared there since you last fetched—safer than a bare `--force`. If you need to **keep** GitHub’s README and merge both histories, ask for help with `git pull github main --allow-unrelated-histories` (you may need to resolve conflicts).

**Login:** GitHub may ask for a username and **password**. For HTTPS, the “password” is usually a **Personal Access Token (PAT)**, not your GitHub account password.

- Create a token: GitHub → **Settings** → **Developer settings** → **Personal access tokens**. Give it **repo** scope for private repos, or **public_repo** for public only (scope names can change; pick access to repositories).
- Copy the token once and paste it when Git asks for a password. Store it in a password manager.

After a successful push, refresh your repository page on GitHub; your files should appear.

---

## Part E — After the first push

- **Actions:** Open your repo on GitHub → tab **Actions**. Workflows in `.github/workflows/` may run automatically (tests, lint, Docker build).
- **Next changes:** After you edit files:

```powershell
git add -A
git commit -m "Short description"
git push github master
```

(Use `main` instead of `master` if that is your branch name.)

---

## Common problems

| What you see | What to try |
|--------------|-------------|
| `remote github already exists` | Use another name (`github2`) or `git remote remove github` and add again (only if you know what you are doing). |
| `failed to push` / `rejected` | Someone else pushed first, or the remote has commits you do not have. For a solo first-time push to an **empty** repo, ensure the GitHub repo was created **without** a README. If needed, ask for help with `git pull --rebase` before pushing. |
| `Authentication failed` | Use a **Personal Access Token** instead of your GitHub password for HTTPS. |
| Huge upload / timeout | Large files may be blocked; keep `exports/`, `data/`, and venv folders **ignored** (see `.gitignore`). |

---

## Your project’s note (extra remote)

This repository may already have a remote (for example **Hugging Face**). Adding **`github`** as in Part D keeps that remote and **adds** GitHub. You push to GitHub with `git push github master` and to the other host with its remote name (for example `git push space master`) if you still use it.

---

## Where to get help

- Official docs: [https://docs.github.com/en/get-started](https://docs.github.com/en/get-started)
- After you push, the **README** in the repo root explains how to run and deploy the app.
