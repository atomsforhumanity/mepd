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

```bash
kubectl apply -f deploy/k8s/mepd-demo.yaml
kubectl -n mepd rollout status deploy/mepd
kubectl -n mepd logs -f deploy/mepd                         # the server's log
kubectl -n mepd port-forward deploy/mepd 8765:8765          # http://localhost:8765, no Ingress needed
```

To deploy a new image, change the `image:` tag and apply again. The old pod
stops before the new one starts (`strategy: Recreate`), so jobs running at
that moment become **interrupted**; testers can resume them.

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
- **Demo limits.** The defaults are in `mepd/web/demo.py` (`DemoPolicy`), and
  any of its numbers can be overridden with `MEPD_DEMO_<FIELD>` in the
  manifest. The manifest runs 64 calculations at once across all testers
  (`MEPD_DEMO_GLOBAL_CONCURRENCY`), and up to 8 per workspace. Each
  calculation uses one core (`OMP_NUM_THREADS=1`). A few operations start 2
  worker processes each, so a full load can briefly ask for more than 64
  cores. Other examples: `MEPD_DEMO_MAX_ATOMS`, `MEPD_DEMO_JOB_TIMEOUT_S`.
  A bad value stops the server at start-up with a message saying which one.
- **The password is the protection.** mepd does not lock anyone out after
  failed logins, because behind the tunnel every visitor can arrive from the
  same address. Use a long random password (e.g. 20+ characters from a
  password manager) in the `mepd-demo` Secret. If you want rate limiting,
  add a Cloudflare rate-limiting rule for `mepd.atomsforhumanity.org/login`;
  Cloudflare sees each visitor's real address.
- **Plain HTTP.** mepd serves plain HTTP. The Cloudflare tunnel terminates
  TLS, and HTTPS redirects and HSTS belong in Cloudflare's settings.
