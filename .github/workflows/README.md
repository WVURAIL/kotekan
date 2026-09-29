# GitHub Actions Workflows

- `build_push_docker.yaml`: Reusable workflow that conditionally builds and publishes the CPU/GPU/Intel Docker images
- `iwyu.yaml`: Manual or scheduled IWYU linter runner
- `main.yaml`: Main CI jobs, builds Docker images and runs the kotekan tests
- `manual_docker.yaml`: Manually trigger Docker image builds
- `publish_docker.yaml`: (Re)Builds and publishes Docker images for PRs or pushes into `develop`
- `schedule.yaml`: Daily cron entry point (runs from the default branch) that dispatches `scheduled_tasks.yaml` and `iwyu.yaml`
- `scheduled_tasks.yaml`: Tasks dispatched by the scheduler on the default branch (`develop`)
- `test_kotekan_build.yaml`: Reusable workflow that builds kotekan and runs post-build commands

## GPU testing

Normal pull requests and pushes run on GitHub-hosted CPU runners. GPU tests
are available through `gpu_tests.yaml` with an explicit confirmation, only
for the repository's default branch. Review and merge the change before
requesting a GPU run; the workflow checks out that selected commit.

Provision an isolated Linux x86-64 runner with the `kotekan-gpu` label,
CUDA/OpenCL support, Docker, and the required `/data` test directory.
The runner must also have the standard `self-hosted`, `Linux`, and `X64` labels.
No matching runner is provisioned by this workflow. Confirm availability
before dispatching; a missing runner leaves the job queued.

These tests require a privileged container and mount `/data`. Keep credentials,
personal files, and sensitive network services away from the runner. Restrict
its runner group to this repository and the workflow that directly defines
the GPU job: `WVURAIL/kotekan/.github/workflows/test_kotekan_build.yaml@refs/heads/develop`.
For another fork, use its repository and default branch. A public pull request
can change workflow YAML, so the workflow's own condition is not a substitute
for this runner-group policy. If that policy cannot be enforced, keep public
repository access to the GPU runner disabled. Do not run unreviewed pull
requests on this hardware. The scheduled context logger uses hosted runners.

GitHub documents workflow/ref restrictions for Enterprise plans; the Team plan
provides repository-level runner-group access. See the
[runner-group documentation](https://docs.github.com/en/enterprise-cloud@latest/actions/how-tos/manage-runners/self-hosted-runners/manage-access#changing-which-workflows-can-access-a-runner-group).
No organization runner policy has been verified here. Keep privileged GPU
access disabled for this public fork unless workflow/ref isolation can be
enforced. Otherwise, use a separate restricted CI repository or run reviewed
commits manually on isolated lab hardware.

`Required CI` is the stable branch-protection check for the image, CPU, Intel,
and lint jobs. It fails if any required job fails, is canceled, or is skipped.
The separate `viewers` check covers browser and Python tooling. GPU tests are
manual and are not prerequisites for routine merges.
