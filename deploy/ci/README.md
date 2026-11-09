# Continuous deployment

`.github/workflows/build-deploy.yml` builds the image, tests it, pushes it to
GHCR and deploys it to the beta cluster:

| Trigger | Image tag | Deployed? |
|---|---|---|
| push (or merge) to `main` | `main-<run>-<sha>`, e.g. `main-14-592a9b4`, plus `main` | yes |
| push a tag `v0.2.0` | `0.2.0` | no |
| Actions → Build and deploy → **Run workflow**, with a tag | (no build) | yes: that existing tag, to redeploy or roll back |

- **The run number** grows with every run, so newer images sort after older ones. **The commit hash**
  says which code an image holds. The moving `main` tag is a convenience; deploys always pin the exact tag.
- **A build** runs the image check (`mepd-entrypoint check`), then the test suite inside the image.
  A failed test is rerun once, alone, because two tests are known to be flaky under parallel load. Only an
  image whose tests pass is pushed.
- **A deploy** applies `deploy/k8s` with that tag, then waits for the new server to be ready. The old server
  keeps serving until then, so the site stays up. If the new one isn't ready within 20 minutes, the deploy is
  rolled back to the previous version and the run fails. It then
  checks that https://mepd.afh.org/login answers. Calculations running at the moment of a
  deploy become *interrupted*, and testers can resume them.

Deploy with the workflow, not by hand. `kubectl apply -k deploy/k8s` from a checkout uses the tag committed
in `deploy/k8s/kustomization.yaml`, which is older than what CI last deployed, so it would roll the cluster
back.

## One-time setup

### 1. The build runner (afh-services, sudo)

The builds run on our own machine, not GitHub's. GitHub's runners have too little disk for an ~18 GB image,
start every build cold, and can't reach the cluster's API on the LAN. The runner goes on **afh-services**,
which is mostly idle and keeps the compute nodes free for science. It runs as your own user, GitHub's default.

**Docker,** usable by you:

```bash
sudo apt-get install -y docker.io docker-buildx git curl   # buildx: BuildKit, for `docker build --secret`
# The build cache trims itself to 300 GB (it would otherwise only grow).
echo '{"builder": {"gc": {"enabled": true, "defaultKeepStorage": "300GB"}}}' | sudo tee /etc/docker/daemon.json
sudo systemctl restart docker
sudo usermod -aG docker "$USER"      # the runner runs as you and needs Docker
```

**The runner,** GitHub's own steps. Go to the **mepd repository → Settings → Actions → Runners → New
self-hosted runner → Linux x64**, and run its *Download* and *Configure* commands in your home folder on
afh-services. When `config.sh` asks, give:
- **runner name:** `afh-services`
- **additional labels:** `mepd-build`. The workflow asks for this label.

Then make it a service that starts with the machine. It runs as you:

```bash
cd ~/actions-runner
sudo ./svc.sh install && sudo ./svc.sh start
sudo ./svc.sh status                 # active (running)
```

The runner then shows as *Idle* on that settings page. The repository is private, so only people with write
access can run jobs on this machine. Those jobs run as you, with Docker access, which amounts to root.

### 2. Cluster access for the runner (afh-services, once)

The deploy step runs `kubectl` on afh-services, using k3s's own copy. Give your user the cluster's
kubeconfig, which points at `127.0.0.1:6443`, i.e. afh-services itself:

```bash
mkdir -p ~/.kube && sudo install -m 600 -o "$USER" /etc/rancher/k3s/k3s.yaml ~/.kube/config
echo 'export KUBECONFIG=~/.kube/config' >> ~/.bashrc && source ~/.bashrc
kubectl get nodes        # works without sudo
```

k3s's `kubectl` reads `/etc/rancher/k3s/k3s.yaml` (root-only) whenever it exists and ignores
`~/.kube/config` unless `KUBECONFIG` names it. The workflow sets `KUBECONFIG` itself; the `.bashrc` line is
for your own use on the machine.

This is the cluster's admin credential. A separate restricted account would protect nothing here: jobs run on
this node with Docker access, which amounts to root, and root can read this file anyway.

### 3. GitHub settings (repository admin)

- **Repository → Settings → Secrets and variables → Actions → New repository secret → `HF_TOKEN`.** This is
  the fine-grained Hugging Face token of `huggingface@atomsforhumanity.org`, with read access to gated repos.
  The build needs it for the UMA weights.
- **`LINUS_DEPLOY_KEY`** (optional): the private half of a read-only deploy key of `atomsforhumanity/linus`
  (linus → Settings → Deploy keys, added by a linus admin; atomsforhumanity/linus#2). With it, each build
  resolves linus's `main` and builds it into the image, and network expansions default to steering with it;
  without it the image has no linus. The key pair was made on Jan's laptop as `~/.ssh/mepd_ci_linus_deploy`:
  `gh secret set LINUS_DEPLOY_KEY -R atomsforhumanity/mepd < ~/.ssh/mepd_ci_linus_deploy`.
- **`LINUS_TOKEN`** (optional, instead of the deploy key): a GitHub token that can read
  `atomsforhumanity/linus` (a fine-grained token with *Contents: read* on that repository only). The deploy
  key is used when both are set.
- Nothing is needed for axtb: it is public, and each build resolves its `main` the same way.
- **The GHCR package** (https://github.com/orgs/atomsforhumanity/packages/container/package/mepd → **Package
  settings → Manage Actions access → Add repository → `mepd`, role *Write***). The first images were pushed by
  hand, so the workflow's built-in token can't push to the package until the repository has access here.

The deploy job runs in a `production` environment, which GitHub creates on the first run. It lists every
deploy under the repository's **Environments**.

### 4. First run

Merge to `main` and watch **Actions → Build and deploy**. The first build on a new runner starts with an
empty cache, so it takes longest: it downloads everything (Hugging Face, the data mirror, conda-forge,
PyPI). Later builds of a code-only change take a few minutes and push one small layer.

## Day to day

- **Deploy:** merge to `main`.
- **Roll back:** Run workflow with an earlier tag. GHCR → package → Versions lists every tag.
- **Watch a deploy:** the run's log, or `kubectl -n mepd rollout status deploy/mepd` and
  `kubectl -n mepd logs deploy/mepd -f`.
- **Runner disk:** the build cache grows. Clear it occasionally on the runner with
  `docker builder prune --filter until=720h`; the next build re-downloads what it removed.
