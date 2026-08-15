# ML Portfolio — Alex Rivera (template)

A personal portfolio site with an ML theme: About, Skills, Projects, Experience,
and Contact sections, built around one centerpiece — a real, working machine
learning model. The Flask backend trains a `RandomForestClassifier` on the
classic Iris dataset at startup, and the "Projects" section embeds a live demo
of it, so a visitor can move sliders and watch real predictions come back from
`/api/predict` instead of looking at a screenshot.

## Files

| File               | Purpose                                                        |
|---------------------|-----------------------------------------------------------------|
| `index.html`         | The whole site — hero, about, skills, projects (incl. live demo), experience, contact |
| `app.py`             | Flask app: trains the model, serves `index.html`, exposes the API |
| `requirements.txt`   | Python dependencies (Flask, scikit-learn, numpy, gunicorn)       |
| `render.yaml`        | Render Blueprint — deploys `app.py` as a web service             |

> Note: Render and pip both require the dependency file to be named exactly
> `requirements.txt` (not `requirement.txt`), so that's the name used here —
> just make sure your repo has it spelled that way.

## Make it yours

`index.html` is filled with placeholder content marked by `<!-- TODO -->`
comments — search for `TODO` in the file to find every spot. In order of
appearance:

1. **Nav & hero** — your name, one-line title/pitch, and the "open to roles"
   badge (or delete it).
2. **Hero links** — GitHub, LinkedIn, email.
3. **About** — bio paragraphs, focus-area tags, and the quick-facts card
   (location, years of experience, current role, resume link).
4. **Skills** — the four tag groups; add/remove/rename as needed.
5. **Projects** — the live demo's source-code link, plus the three static
   project cards (swap in your real projects, stacks, and links). You can
   also replace the Iris classifier itself with your own model — see below.
6. **Experience** — the timeline entries.
7. **Contact** — email, GitHub, LinkedIn.
8. **`<title>` and meta description** at the top of the file.

### Swapping in your own model

The live demo doesn't have to be Iris. To replace it:
- In `app.py`, swap the `load_iris()` / training block for your own dataset
  and model, and adjust `FEATURE_RANGES` and the fields in `/api/predict`
  to match your model's inputs.
- In `index.html`, update the slider labels in `buildSliders()` and the
  `COLORS` map to match your output classes (or restyle the result panel
  entirely if your model isn't a classifier — e.g. show a single predicted
  number instead of a probability bar chart).

## API

- `GET /` — serves the site
- `GET /api/meta` — species list, slider ranges, dataset points for the scatter plot, test accuracy
- `POST /api/predict` — body `{"sepal_length": 5.8, "sepal_width": 3.0, "petal_length": 4.3, "petal_width": 1.3}` → returns predicted species + probabilities
- `GET /api/health` — quick status check

## Run it locally

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Visit `http://localhost:5000`.

## Deploy: GitHub → Render → Hostinger domain

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "ML site: Iris classifier"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

### 2. Deploy on Render

1. Go to [render.com](https://render.com) → **New** → **Blueprint**.
2. Connect your GitHub account and select this repo. Render will read
   `render.yaml` automatically and configure the web service (build command,
   start command, Python version) for you.
3. Click **Apply**. First deploy takes a couple of minutes — Render installs
   `requirements.txt` and starts `gunicorn app:app`.
4. Once live, you'll get a URL like `https://ml-iris-classifier.onrender.com`.
   Confirm it works, including `/api/predict`.

If you'd rather not use the Blueprint, you can create the service manually:
**New → Web Service**, connect the repo, set:
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`

### 3. Point your Hostinger domain at Render

Render (not Hostinger) hosts the actual app, so the domain just needs to point
there:

1. In the Render dashboard, open your service → **Settings → Custom Domains**
   → **Add Custom Domain**. Enter your domain, e.g. `ml.yourdomain.com` (a
   subdomain) or `yourdomain.com` (apex).
2. Render will show you a DNS target — usually a **CNAME** value like
   `ml-iris-classifier.onrender.com` for a subdomain.
3. In **Hostinger → Domains → DNS / Name Servers → DNS Zone Editor**, add:
   - **Subdomain** (recommended, e.g. `ml`): a `CNAME` record —
     Name: `ml`, Points to: the value Render gave you, TTL: default.
   - **Apex domain** (`yourdomain.com` with no subdomain): Hostinger doesn't
     allow a CNAME at the root, so use the `A` record(s) Render provides
     instead, or set up an `ALIAS`/`ANAME` record if your Hostinger plan
     supports it.
4. Save, then wait for DNS to propagate (usually minutes, sometimes up to a
   few hours). Render will show the domain status as "Verified" once it sees
   the correct DNS record, and will auto-provision an SSL certificate.
5. Visit your domain — it should now load the site served from Render.

### Notes

- Render's free plan spins the service down after periods of inactivity;
  the first request after idling can take ~30–50 seconds to wake up.
- Every `git push` to `main` triggers an automatic redeploy (`autoDeploy: true`
  in `render.yaml`).
- The model retrains from scratch on every server start (it's fast — Iris is
  150 rows), so there's no model file to commit or manage.
