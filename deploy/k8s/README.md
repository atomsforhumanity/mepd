# mepd on the afh cluster

`mepd-demo.yaml` runs the image (`ghcr.io/atomsforhumanity/mepd`) in demo
mode on **afh-c04**. Every tester shares one password and gets private
workspaces, with the limits in `mepd/web/demo.py`. It runs as one pod with
the node's GPU, and keeps its state on the node's data disk (`/nvme1`).

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
# The testers' shared password
kubectl -n mepd create secret generic mepd-demo --from-literal=password='<password>'
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

A new version stops the old pod before starting the new one (`strategy: Recreate`), so jobs running at that
moment become **interrupted**; testers can resume them.

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

- **One replica, always.** The server holds the job queue in memory and
  manages each workspace's jobs. Two servers would both manage them.
- **The GPU.** The pod takes the node's one GPU (`nvidia.com/gpu: 1`) for
  the machine-learned potentials. g-xTB, xtb, CREST and Psi4 run on CPU.
- **Demo limits.** The defaults in `mepd/web/demo.py` (`DemoPolicy`) suit a
  small shared demo. The manifest sizes them for this node with
  `MEPD_DEMO_<FIELD>` variables:
  - 64 calculations at once, 8 per workspace;
  - 150 atoms per structure;
  - 4 h per calculation;
  - 8 workers per operation and per profile;
  - 8 interactive reactors;
  - QM/MM systems up to 50,000 atoms.

  Numbers are plain values (`MEPD_DEMO_MAX_ATOMS=150`). Tables (`CAPS`,
  `OP_TIMEOUT_S`, `PROFILE_CAPS`, `SANDBOX`, `QMMM`) are JSON merged into the
  defaults, e.g. `MEPD_DEMO_CAPS='{"channels": {"workers": 16}}'`. A bad value
  or a misspelled name stops the server at start-up with a message saying
  which. What may run at all (operations, engines, forbidden settings) cannot
  be changed from the manifest.
- **What testers can't do** (by design): read files on the server, run the
  ASE, QCCompute or ChemCloud engines (they run arbitrary code or programs),
  set programs, file paths or credentials in a profile, use a local-LLM
  retrosynthesis server, or ask Psi4 for more than 16 GB per job
  (`MEPD_DEMO_PSI4_MAX_MEMORY_GB`). Everything else in the app is available.
- **The password is the protection.** mepd does not lock anyone out after
  failed logins, because behind the tunnel every visitor can arrive from the
  same address. Use a long random password (e.g. 20+ characters from a
  password manager) in the `mepd-demo` Secret. If you want rate limiting,
  add a Cloudflare rate-limiting rule for `mepd.atomsforhumanity.org/login`;
  Cloudflare sees each visitor's real address.
- **Plain HTTP.** mepd serves plain HTTP. The Cloudflare tunnel terminates
  TLS, and HTTPS redirects and HSTS belong in Cloudflare's settings.
