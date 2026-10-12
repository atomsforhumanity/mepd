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
- **A deploy** applies `deploy/k8s` with that tag, then waits for the new server to be ready. If it isn't
  ready within 20 minutes, the deploy is rolled back to the previous version and the run fails. It then
  checks that https://mepd.atomsforhumanity.org/login answers. Calculations running at the moment of a
  deploy become *interrupted*, and testers can resume them.

Deploy with the workflow, not by hand. `kubectl apply -k deploy/k8s` from a checkout uses the tag committed
in `deploy/k8s/kustomization.yaml`, which is older than what CI last deployed, so it would roll the cluster
back.

## One-time setup

### 1. The build runner (afh-services, sudo)

The builds run on our own machine, not GitHub's. GitHub's runners have too little disk for an ~18 GB image,
start every build cold, and can't reach the cluster's API on the LAN. The runner goes on **afh-services**,
which keeps the compute nodes free for science. afh-services also runs the whole public path (the Cloudflare
tunnel, Traefik, the k3s control plane), so builds run in a **low-priority slice with a memory cap**. When
anything else needs the machine, the build yields; when the machine is idle, the build gets all of it.

**Check the uplink first.** After a dependency change, the image's 7.4 GB Python layer must upload to GHCR
within GHCR's 10-minute limit per upload, which needs a sustained ~100 Mbit/s or more:

```bash
head -c 200000000 /dev/urandom | curl -s -o /dev/null -w '%{speed_upload}\n' --data-binary @- https://speed.cloudflare.com/__up
# bytes/s: 12500000 or more is 100 Mbit/s
```

**The slice, Docker and its configuration:**

```bash
sudo apt-get install -y docker.io docker-buildx git curl   # buildx: BuildKit, for `docker build --secret`

# ci.slice: lowest CPU and disk priority (k3s, Traefik, cloudflared win any contest),
# and at most 40 of the 62 GB, so a build can never take the web services' memory.
sudo tee /etc/systemd/system/ci.slice >/dev/null <<'UNIT'
[Unit]
Description=CI builds (low priority: the cluster and web services come first)
[Slice]
CPUWeight=20
IOWeight=20
MemoryHigh=36G
MemoryMax=40G
UNIT

# Docker and its containerd run in it. (k3s has its own containerd; it is not affected.)
for svc in docker containerd; do
  sudo mkdir -p /etc/systemd/system/$svc.service.d
  printf '[Service]\nSlice=ci.slice\n' | sudo tee /etc/systemd/system/$svc.service.d/ci-slice.conf >/dev/null
done

# Containers and build steps in it too; the build cache trims itself to 300 GB.
echo '{"cgroup-parent": "ci.slice", "builder": {"gc": {"enabled": true, "defaultKeepStorage": "300GB"}}}' \
  | sudo tee /etc/docker/daemon.json
sudo systemctl daemon-reload && sudo systemctl restart containerd docker

# A user for the runner, allowed to use Docker. Docker access amounts to root on this machine;
# the repository is private, so only people with write access can run jobs here.
sudo useradd --create-home --shell /bin/bash github-runner
sudo usermod -aG docker github-runner
```

**Register the runner.** On GitHub, go to the **mepd repository → Settings → Actions → Runners → New
self-hosted runner → Linux x64**. Run the download commands it shows as `github-runner`, then:

```bash
sudo -iu github-runner
# ... the download/extract commands GitHub shows (they create ~/actions-runner), then:
cd ~/actions-runner
./config.sh --url https://github.com/atomsforhumanity/mepd --token <shown on that page> \
  --name afh-services --labels mepd-build --unattended
exit

cd /home/github-runner/actions-runner
sudo ./svc.sh install github-runner
UNIT=$(cat .service)                       # actions.runner.<org>-<repo>.afh-services.service
sudo mkdir -p /etc/systemd/system/$UNIT.d
printf '[Service]\nSlice=ci.slice\n' | sudo tee /etc/systemd/system/$UNIT.d/ci-slice.conf >/dev/null
sudo systemctl daemon-reload && sudo ./svc.sh start
```

The label `mepd-build` is what the workflow asks for, and the runner then shows as *Idle* on that settings
page.

**Check the guardrails during the first build:**
- `systemd-cgls /ci.slice` lists the runner, Docker, and the build and test processes.
- `systemctl status ci.slice` shows the slice's memory against its 40 GB cap.

### 2. The deploy account (a cluster admin, once)

```bash
kubectl apply -f deploy/k8s/ci-deployer.yaml
```

This creates `ci-deployer`, an account that can only change mepd's Deployment, Service and Ingress in the
`mepd` namespace, and gives it a long-lived token. Then make a kubeconfig for it:

```bash
SERVER=https://192.168.8.6:6443      # the k3s API (afh-services itself, where the runner is)
CA=$(kubectl -n mepd get secret ci-deployer-token -o jsonpath='{.data.ca\.crt}')
TOKEN=$(kubectl -n mepd get secret ci-deployer-token -o jsonpath='{.data.token}' | base64 -d)
cat > ci-deployer.kubeconfig <<EOF
apiVersion: v1
kind: Config
clusters: [{name: afh, cluster: {server: $SERVER, certificate-authority-data: $CA}}]
users: [{name: ci-deployer, user: {token: $TOKEN}}]
contexts: [{name: ci, context: {cluster: afh, user: ci-deployer, namespace: mepd}}]
current-context: ci
EOF
chmod 600 ci-deployer.kubeconfig
KUBECONFIG=ci-deployer.kubeconfig kubectl -n mepd get deploy mepd      # works
KUBECONFIG=ci-deployer.kubeconfig kubectl get nodes                    # forbidden, as intended
```

Its contents go into the `KUBE_CONFIG` secret (step 3). Then delete the file. To revoke the token,
`kubectl -n mepd delete secret ci-deployer-token`; re-applying the YAML makes a new one.

### 3. GitHub settings (repository admin)

In the **mepd repository → Settings**:

- **Environments → New environment → `production`.** The deploy job runs in it, and GitHub keeps a deploy
  history there. In that environment, add the secret **`KUBE_CONFIG`**: the whole kubeconfig file from
  step 2.
- **Secrets and variables → Actions → New repository secret → `HF_TOKEN`.** This is the fine-grained Hugging
  Face token of `huggingface@atomsforhumanity.org`, with read access to gated repos. The build needs it for
  the UMA weights.

In the **GHCR package** (https://github.com/orgs/atomsforhumanity/packages/container/package/mepd → **Package settings**):

- **Manage Actions access → Add repository → `mepd`, role *Write*.** The first images were pushed by hand,
  so the workflow's built-in token can't push to the package until the repository is given access here.

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
