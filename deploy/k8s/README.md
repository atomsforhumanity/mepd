# mepd on the afh cluster

`mepd-demo.yaml` runs the image (`ghcr.io/atomsforhumanity/mepd`) in demo
mode on **afh-c04**. It is open: anyone who opens the page gets private
workspaces at once, no password, with the limits in `mepd/web/demo.py`. It runs as one pod that
uses the node's GPU, and keeps its state on the node's data disk (`/nvme1`).

## One-time setup

```bash
# On afh-c04: the data disk directories, owned by the container's user (uid 1000)
sudo mkdir -p /nvme1/mepd/data /nvme1/mepd/tmp
sudo chown -R 1000:1000 /nvme1/mepd

# From a machine with kubectl:
kubectl create namespace mepd
# Pull access to the private image: a classic GitHub PAT with only read:packages,
# ideally from a machine account rather than a person's.
kubectl -n mepd create secret docker-registry ghcr-pull --docker-server=ghcr.io \
  --docker-username=<github user> --docker-password=<token>
```

## Deploy, update, inspect

**Deploys are done by CI.** A merge to `main` builds, tests and deploys a new image
(`.github/workflows/build-deploy.yml`, set up in `deploy/ci/README.md`). To redeploy or roll back to an
existing image, use the workflow's **Run workflow** button with its tag. The image tag is set at deploy
time in `kustomization.yaml`; the tag committed there is only an older fallback, so a
`kubectl apply -k deploy/k8s` by hand would roll the cluster back to it.

```bash
kubectl -n mepd rollout status deploy/mepd
kubectl -n mepd logs -f deploy/mepd                         # the server's log
kubectl -n mepd get deploy mepd -o jsonpath='{..image}'     # which image is live
kubectl -n mepd port-forward deploy/mepd 8765:8765          # http://localhost:8765, no Ingress needed
```

A new version starts beside the old one, and the old pod stops only once the new one is ready
(`strategy: RollingUpdate`), so the site stays up. Jobs running in the old pod become **interrupted**;
testers can resume them. For the minute both pods run, the new server already shows those jobs as
interrupted, and a job that finishes in that minute still shows as interrupted until the next restart.

## Where things are

| On afh-c04 | In the container | What |
|---|---|---|
| `/nvme1/mepd/data/demo/visitors/<id>/` | `/data/demo/visitors/<id>/` | each tester's workspaces and job outputs |
| `/nvme1/mepd/data/demo/profiles/*.toml` | `/data/demo/profiles/` | the compute profiles testers run with (read-only for them; edit here) |
| `/nvme1/mepd/data/state/` | `/data/state/` | logins and sessions |
| `/nvme1/mepd/data/demo/.demo_secret` | | cookie-signing key: delete it to log everyone out |
| `/nvme1/mepd/tmp/` | `/tmp` | scratch (GSM, Psi4); safe to clear while the pod is stopped |

Back up `/nvme1/mepd/data`. Everything else can be rebuilt.

## Notes

- **One replica.** The server holds the job queue in memory and manages
  each workspace's jobs. Two servers run together only during a deploy.
- **Resources are barely reserved.** The pod requests 2 cores and 8 GB but
  can use all 64 cores and up to 224 GB, so the node stays open to other work
  and may be oversubscribed.
- **The GPU.** The pod sees the node's GPU (`NVIDIA_VISIBLE_DEVICES=all`)
  without reserving it (no `nvidia.com/gpu`), so the new pod of a deploy can
  start beside the old one, and other pods can use it too. It's for the
  machine-learned potentials; g-xTB, xtb, CREST and Psi4 run on CPU.
- **Demo limits.** Since 2026-11-10 the demo does not limit compute
  (`DemoPolicy.limit_compute` off in `mepd/web/demo.py`): no caps on atoms,
  structures, operation settings, run times or a profile's parallelism. The
  manifest keeps only what queues work: 64 calculations at once across all
  testers, 8 per workspace (the rest wait), and 8 interactive reactors. To
  restore the limits, set `limit_compute = True`; they can then be sized with
  `MEPD_DEMO_<FIELD>` variables: numbers plain (`MEPD_DEMO_MAX_ATOMS=150`),
  tables (`CAPS`, `OP_TIMEOUT_S`, `PROFILE_CAPS`, `SANDBOX`, `QMMM`) as JSON
  merged into the defaults. A bad value or a misspelled name stops the server
  at start-up with a message saying which. What may run at all (operations,
  engines, forbidden settings) cannot be changed from the manifest.
- **What testers can't do** (by design): read files on the server, run the
  ASE, QCCompute or ChemCloud engines (they run arbitrary code or programs),
  set programs, file paths or credentials in a profile, use a local-LLM
  retrosynthesis server. Everything else in the app is available.
- **No password and no compute limits.** Anyone on the internet can use the
  demo's compute, within the demo's blocks on programs, paths and credentials
  and the queue above. A new visitor is made by opening the page,
  and their workspace by the page's first API call, so crawlers that run no
  JavaScript leave nothing on disk. To slow abuse, add a Cloudflare
  rate-limiting rule for `mepd.afh.org/` and `/api/jobs`;
  Cloudflare sees each visitor's real address. To require a shared password
  again, set `MEPD_DEMO_PASSWORD` (e.g. from a Secret) in `mepd-demo.yaml`.
- **Plain HTTP.** mepd serves plain HTTP. The Cloudflare tunnel terminates
  TLS, and HTTPS redirects and HSTS belong in Cloudflare's settings.
