#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/.openvino_install_work"

OS_ID=""
OS_VERSION=""

# --prefix mode. The default path here installs GPU and NPU drivers as root and
# drops the runtime in /opt/intel, which is the right thing on a machine you own
# and impossible on a shared cluster: no sudo, no apt, and the drivers already
# provided by the site (on ours, /swtools/intel-gpu). --prefix fetches and
# unpacks *only* the runtime archive, into a directory you can write.
#
# The archive is still Ubuntu-built, because that is the only flavour Intel
# publishes; it runs on other glibc-compatible distributions, which is why
# --ubuntu exists to choose one when /etc/os-release is not Ubuntu at all.
PREFIX=""
SUDO="sudo"
UBUNTU_OVERRIDE=""

usage() {
  cat <<'EOF'
install_ov.sh -- OpenVINO runtime, and on Ubuntu the GPU/NPU drivers too.

  install_ov.sh                        full install: drivers + runtime in
                                       /opt/intel, needs sudo, Ubuntu only
  install_ov.sh --prefix ~/intel       runtime only, no sudo, no apt, any distro
  install_ov.sh --prefix ~/intel --ubuntu 24.04
                                       same, choosing the archive flavour when
                                       the host is not Ubuntu

Options:
  --prefix DIR     install the runtime under DIR (implies runtime-only, no sudo)
  --ubuntu VER     22.04 or 24.04; which archive to fetch. Required with
                   --prefix on a non-Ubuntu host.
  -h, --help       this

Environment: OPENVINO_VERSION, OPENVINO_BUILD, INSTALL_ROOT (overridden by
--prefix). With a version override the mirror's published checksum is used
instead of the digest pinned in this script.

After a --prefix install:  source DIR/openvino/setupvars.sh
EOF
}

log() {
  printf '[install_openvino_runtime] %s\n' "$*"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Error: '$1' is required but not installed." >&2
    exit 1
  }
}

# Digests of the two archives the defaults below pin. These land in /opt under
# sudo, so a bad download is a root-level problem.
OPENVINO_SHA256_2204="d701a115d3dc18088ff75b5b8e67a51fbf780022a3d40ee8ee7f2adfbd9915e6"
OPENVINO_SHA256_2404="6931e5a3c9b1fc9cb170137196df2c40489625703f2d184f511b7add2c110ef8"

# verify_sha256 <file> <expected-or-empty> <url>. An overridden version has no
# digest here, so fall back to the one the mirror publishes: that catches a
# truncated or corrupted download, not a compromised mirror.
verify_sha256() {
  local path="$1" want="$2" url="$3"
  if [[ -z "${want}" ]]; then
    want="$(curl -fsSL "${url}.sha256" | awk 'NR==1 {print $1}')"
    if [[ ! "${want}" =~ ^[0-9a-f]{64}$ ]]; then
      echo "Error: no usable checksum published for ${url}" >&2
      exit 1
    fi
    log "Version overridden, using the mirror's own checksum."
  fi
  if ! printf '%s  %s\n' "${want}" "${path}" | sha256sum -c - >/dev/null; then
    echo "Error: checksum mismatch for ${path}" >&2
    exit 1
  fi
  log "Checksum OK: $(basename "${path}")"
}

detect_os() {
  if [[ ! -f /etc/os-release ]]; then
    echo "Error: /etc/os-release not found; cannot detect Ubuntu version." >&2
    exit 1
  fi

  # shellcheck disable=SC1091
  source /etc/os-release

  OS_ID="${ID:-}"
  OS_VERSION="${VERSION_ID:-}"

  if [[ "${OS_ID}" != "ubuntu" ]]; then
    echo "Error: this installer supports Ubuntu only. Detected ID='${OS_ID:-unknown}'." >&2
    exit 1
  fi

  # Every package name below is amd64. Say so now, not after the first 404.
  local arch
  arch="$(uname -m)"
  if [[ "${arch}" != "x86_64" ]]; then
    echo "Error: Intel publishes these packages for x86_64 only. Detected '${arch}'." >&2
    exit 1
  fi
}

prepare_common_tools() {
  need_cmd bash
  need_cmd curl
  need_cmd tar
  need_cmd sha256sum

  # Only the driver-package path needs these, and demanding them in --prefix
  # mode would fail on exactly the hosts --prefix exists for.
  if [[ -z "${PREFIX}" ]]; then
    need_cmd sudo
    need_cmd apt-get
    need_cmd wget
    need_cmd dpkg
    need_cmd find
    need_cmd sort
  fi

  mkdir -p "${WORK_DIR}"
}

prepare_common_dependencies() {
  log "Installing common dependencies..."
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    libcurl4-openssl-dev \
    libtbb12 \
    cmake \
    ninja-build \
    python3-pip \
    curl \
    wget \
    tar \
    libopencl1 \
    ocl-icd-opencl-dev \
    opencl-headers \
    opencl-clhpp-headers \
    clinfo
}

add_render_group() {
  # USER is unbound, not empty, in a container or a cron shell, and set -u would
  # abort here with the drivers already installed.
  local target_user="${SUDO_USER:-${USER:-}}"
  if [[ -z "${target_user}" ]]; then
    return
  fi

  if id -nG "${target_user}" | grep -qw render; then
    log "User ${target_user} is already in the render group."
  else
    log "Adding ${target_user} to render group..."
    sudo gpasswd -a "${target_user}" render || true
    log "Re-login (or restart shell session) to apply render group membership."
  fi
}

install_gpu_2204() {
  local download_dir="${WORK_DIR}/intel_gpu_2204"
  local igc_base_url="https://github.com/intel/intel-graphics-compiler/releases/download/v2.10.8"
  local crt_base_url="https://github.com/intel/compute-runtime/releases/download/25.13.33276.16"
  local checksum_file="ww13.sum"
  local packages=(
    "${igc_base_url}/intel-igc-core-2_2.10.8+18926_amd64.deb"
    "${igc_base_url}/intel-igc-opencl-2_2.10.8+18926_amd64.deb"
    "${crt_base_url}/intel-level-zero-gpu-dbgsym_1.6.33276.16_amd64.ddeb"
    "${crt_base_url}/intel-level-zero-gpu_1.6.33276.16_amd64.deb"
    "${crt_base_url}/intel-opencl-icd-dbgsym_25.13.33276.16_amd64.ddeb"
    "${crt_base_url}/intel-opencl-icd_25.13.33276.16_amd64.deb"
    "${crt_base_url}/libigdgmm12_22.7.0_amd64.deb"
  )

  log "Installing Intel GPU drivers for Ubuntu 22.04..."
  mkdir -p "${download_dir}"
  cd "${download_dir}"

  for url in "${packages[@]}"; do
    wget --no-continue "${url}"
  done
  wget --no-continue "${crt_base_url}/${checksum_file}"

  sha256sum --ignore-missing -c "${checksum_file}"

  shopt -s nullglob
  local artifacts=( *.deb *.ddeb )
  shopt -u nullglob
  if [[ ${#artifacts[@]} -eq 0 ]]; then
    echo "Error: no GPU package files found for Ubuntu 22.04." >&2
    exit 1
  fi

  sudo dpkg -i "${artifacts[@]}" || sudo apt-get install -f -y
  cd "${SCRIPT_DIR}"
  rm -rf "${download_dir}"
}

install_npu_2204() {
  local download_dir="${WORK_DIR}/intel_npu_2204"
  local npu_tarball="linux-npu-driver-v1.26.0.20251125-19665715237-ubuntu2204.tar.gz"
  local npu_url="https://github.com/intel/linux-npu-driver/releases/download/v1.26.0/${npu_tarball}"
  local level_zero_deb="level-zero_1.24.2+u22.04_amd64.deb"
  local level_zero_url="https://github.com/oneapi-src/level-zero/releases/download/v1.24.2/${level_zero_deb}"

  log "Installing Intel NPU drivers for Ubuntu 22.04..."
  mkdir -p "${download_dir}"
  cd "${download_dir}"

  # Download before purging. The other order leaves a machine with no NPU driver
  # at all if the fetch fails.
  wget --no-continue "${npu_url}"
  wget --no-continue "${level_zero_url}"
  tar -xf "${npu_tarball}"

  mapfile -t npu_debs < <(find . -type f -name '*.deb' ! -name 'level-zero*.deb' | sort)
  if [[ ${#npu_debs[@]} -eq 0 ]]; then
    echo "Error: no Intel NPU .deb packages found for Ubuntu 22.04." >&2
    exit 1
  fi

  sudo dpkg --purge --force-remove-reinstreq \
    intel-driver-compiler-npu \
    intel-fw-npu \
    intel-level-zero-npu \
    intel-level-zero-npu-dbgsym || true

  sudo dpkg -i "${npu_debs[@]}" || sudo apt-get install -f -y
  sudo dpkg -i "${level_zero_deb}" || sudo apt-get install -f -y

  add_render_group

  cd "${SCRIPT_DIR}"
  rm -rf "${download_dir}"
}

install_runtime_2204() {
  local download_dir="${WORK_DIR}/openvino_runtime_2204"
  local openvino_version="${OPENVINO_VERSION:-2025.3}"
  local openvino_build="${OPENVINO_BUILD:-19807.44526285f24}"
  local openvino_archive="openvino_toolkit_ubuntu22_${openvino_version}.0.${openvino_build}_x86_64.tgz"
  local openvino_dirname="openvino_toolkit_ubuntu22_${openvino_version}.0.${openvino_build}_x86_64"
  local openvino_url="https://storage.openvinotoolkit.org/repositories/openvino/packages/${openvino_version}/linux/${openvino_archive}"
  local install_root="${INSTALL_ROOT:-/opt/intel}"
  local install_dir="${install_root}/openvino_${openvino_version}"
  local symlink_path="${install_root}/openvino"
  local archive_path="${download_dir}/openvino_${openvino_version}.tgz"
  local expected_sha=""
  if [[ "${openvino_version}" == "2025.3" && "${openvino_build}" == "19807.44526285f24" ]]; then
    expected_sha="${OPENVINO_SHA256_2204}"
  fi

  log "Installing OpenVINO runtime for Ubuntu 22.04..."
  ${SUDO} mkdir -p "${install_root}"
  mkdir -p "${download_dir}"

  curl -fL "${openvino_url}" --output "${archive_path}"
  verify_sha256 "${archive_path}" "${expected_sha}" "${openvino_url}"
  rm -rf "${download_dir:?}/${openvino_dirname}"
  tar -xf "${archive_path}" -C "${download_dir}"

  ${SUDO} rm -rf "${install_dir}"
  ${SUDO} mv "${download_dir}/${openvino_dirname}" "${install_dir}"
  ${SUDO} ln -sfn "openvino_${openvino_version}" "${symlink_path}"

  if [[ ! -f "${symlink_path}/setupvars.sh" ]]; then
    echo "Error: ${symlink_path}/setupvars.sh was not found after installation." >&2
    exit 1
  fi

  rm -rf "${download_dir}"
}

install_gpu_2404() {
  local download_dir="${WORK_DIR}/intel_gpu_2404"
  local igc_base_url="https://github.com/intel/intel-graphics-compiler/releases/download/v2.36.3"
  local crt_base_url="https://github.com/intel/compute-runtime/releases/download/26.22.38646.4"
  local checksum_file="ww22.sum"
  local packages=(
    "${igc_base_url}/intel-igc-core-2_2.36.3+21719_amd64.deb"
    "${igc_base_url}/intel-igc-opencl-2_2.36.3+21719_amd64.deb"
    "${crt_base_url}/intel-ocloc-dbgsym_26.22.38646.4-0_amd64.ddeb"
    "${crt_base_url}/intel-ocloc_26.22.38646.4-0_amd64.deb"
    "${crt_base_url}/intel-opencl-icd-dbgsym_26.22.38646.4-0_amd64.ddeb"
    "${crt_base_url}/intel-opencl-icd_26.22.38646.4-0_amd64.deb"
    "${crt_base_url}/libigdgmm12_22.10.0_amd64.deb"
    "${crt_base_url}/libze-intel-gpu1-dbgsym_26.22.38646.4-0_amd64.ddeb"
    "${crt_base_url}/libze-intel-gpu1_26.22.38646.4-0_amd64.deb"
  )

  log "Installing Intel GPU drivers for Ubuntu 24.04..."
  mkdir -p "${download_dir}"
  cd "${download_dir}"

  for url in "${packages[@]}"; do
    wget --no-continue "${url}"
  done
  wget --no-continue "${crt_base_url}/${checksum_file}"

  sha256sum --ignore-missing -c "${checksum_file}"

  shopt -s nullglob
  local artifacts=( *.deb *.ddeb )
  shopt -u nullglob
  if [[ ${#artifacts[@]} -eq 0 ]]; then
    echo "Error: no GPU package files found for Ubuntu 24.04." >&2
    exit 1
  fi

  sudo dpkg -i "${artifacts[@]}" || sudo apt-get install -f -y
  cd "${SCRIPT_DIR}"
  rm -rf "${download_dir}"
}

install_npu_2404() {
  local download_dir="${WORK_DIR}/intel_npu_2404"
  local npu_release="v1.33.0"
  local npu_archive="linux-npu-driver-v1.33.0.20260529-26625960453-ubuntu2404.tar.gz"
  local npu_url="https://github.com/intel/linux-npu-driver/releases/download/${npu_release}/${npu_archive}"
  local npu_packages=(
    intel-driver-compiler-npu
    intel-fw-npu
    intel-level-zero-npu
    intel-level-zero-npu-dbgsym
  )

  log "Installing Intel NPU drivers for Ubuntu 24.04..."
  mkdir -p "${download_dir}"
  cd "${download_dir}"

  # Download before purging, so a failed fetch does not leave the machine with
  # no NPU driver at all.
  wget --no-continue "${npu_url}"
  tar -xf "${npu_archive}"

  shopt -s nullglob
  local debs=( *.deb )
  shopt -u nullglob
  if [[ ${#debs[@]} -eq 0 ]]; then
    echo "Error: no Intel NPU .deb packages found for Ubuntu 24.04." >&2
    exit 1
  fi

  sudo dpkg --purge --force-remove-reinstreq "${npu_packages[@]}" || true
  sudo dpkg -i "${debs[@]}" || sudo apt-get install -f -y

  # The Level Zero loader the NPU plugin dlopens. 22.04 needs it from GitHub;
  # 24.04 has it in the archive. See docs/backend/ov.md for ZE_ENABLE_ALT_DRIVERS.
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y libze1

  add_render_group

  cd "${SCRIPT_DIR}"
  rm -rf "${download_dir}"
}

install_runtime_2404() {
  local download_dir="${WORK_DIR}/openvino_runtime_2404"
  local openvino_version="${OPENVINO_VERSION:-2026.2.1}"
  local openvino_build="${OPENVINO_BUILD:-21919.ede283a88e3}"
  local openvino_archive="openvino_toolkit_ubuntu24_${openvino_version}.${openvino_build}_x86_64.tgz"
  local openvino_dirname="openvino_toolkit_ubuntu24_${openvino_version}.${openvino_build}_x86_64"
  local openvino_url="https://storage.openvinotoolkit.org/repositories/openvino/packages/${openvino_version}/linux/${openvino_archive}"
  local install_root="${INSTALL_ROOT:-/opt/intel}"
  local install_dir="${install_root}/openvino_${openvino_version}"
  local symlink_path="${install_root}/openvino"
  local archive_path="${download_dir}/openvino_${openvino_version}.tgz"
  local expected_sha=""
  if [[ "${openvino_version}" == "2026.2.1" && "${openvino_build}" == "21919.ede283a88e3" ]]; then
    expected_sha="${OPENVINO_SHA256_2404}"
  fi

  log "Installing OpenVINO runtime for Ubuntu 24.04..."
  ${SUDO} mkdir -p "${install_root}"
  mkdir -p "${download_dir}"

  curl -fL "${openvino_url}" --output "${archive_path}"
  verify_sha256 "${archive_path}" "${expected_sha}" "${openvino_url}"
  rm -rf "${download_dir:?}/${openvino_dirname}"
  tar -xf "${archive_path}" -C "${download_dir}"

  ${SUDO} rm -rf "${install_dir}"
  ${SUDO} mv "${download_dir}/${openvino_dirname}" "${install_dir}"
  ${SUDO} ln -sfn "openvino_${openvino_version}" "${symlink_path}"

  if [[ ! -f "${symlink_path}/setupvars.sh" ]]; then
    echo "Error: ${symlink_path}/setupvars.sh was not found after installation." >&2
    exit 1
  fi

  rm -rf "${download_dir}"
}

run_installation() {
  case "${OS_VERSION}" in
    22.04)
      if [[ -z "${PREFIX}" ]]; then
        install_gpu_2204
        install_npu_2204
      fi
      install_runtime_2204
      ;;
    24.04)
      if [[ -z "${PREFIX}" ]]; then
        install_gpu_2404
        install_npu_2404
      fi
      install_runtime_2404
      ;;
    *)
      echo "Error: unsupported Ubuntu version '${OS_VERSION:-unknown}'. Supported versions: 22.04, 24.04." >&2
      exit 1
      ;;
  esac
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --prefix)
        [[ $# -ge 2 ]] || { echo "Error: --prefix needs a directory." >&2; exit 1; }
        PREFIX="$2"
        shift 2
        ;;
      --prefix=*)
        PREFIX="${1#*=}"
        shift
        ;;
      --ubuntu)
        [[ $# -ge 2 ]] || { echo "Error: --ubuntu needs 22.04 or 24.04." >&2; exit 1; }
        UBUNTU_OVERRIDE="$2"
        shift 2
        ;;
      --ubuntu=*)
        UBUNTU_OVERRIDE="${1#*=}"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "Error: unknown argument '$1'." >&2
        usage >&2
        exit 1
        ;;
    esac
  done

  if [[ -n "${PREFIX}" ]]; then
    # mkdir now rather than at unpack time: a prefix you cannot write is the
    # likeliest way to use this flag wrong, and finding out after a 400 MB
    # download is a poor way to be told.
    mkdir -p "${PREFIX}" || { echo "Error: cannot create --prefix '${PREFIX}'." >&2; exit 1; }
    PREFIX="$(cd "${PREFIX}" && pwd)"
    INSTALL_ROOT="${PREFIX}"
    SUDO=""
  elif [[ -n "${UBUNTU_OVERRIDE}" ]]; then
    echo "Error: --ubuntu only applies with --prefix; the driver packages are" >&2
    echo "       matched to the running system and cannot be chosen." >&2
    exit 1
  fi
}

main() {
  parse_args "$@"

  if [[ -n "${PREFIX}" ]]; then
    # No apt, no /etc/os-release contract: only the archive flavour matters.
    OS_VERSION="${UBUNTU_OVERRIDE:-}"
    if [[ -z "${OS_VERSION}" ]]; then
      if [[ -f /etc/os-release ]]; then
        # shellcheck disable=SC1091
        source /etc/os-release
        [[ "${ID:-}" == "ubuntu" ]] && OS_VERSION="${VERSION_ID:-}"
      fi
    fi
    if [[ -z "${OS_VERSION}" ]]; then
      echo "Error: not running Ubuntu, so the archive flavour cannot be inferred." >&2
      echo "       Pass --ubuntu 22.04 or --ubuntu 24.04." >&2
      exit 1
    fi
    prepare_common_tools
    log "Runtime-only install under ${PREFIX} (Ubuntu ${OS_VERSION} archive, no sudo)."
    run_installation
    rm -rf "${WORK_DIR}"
    log "OpenVINO runtime installed."
    log "To load OpenVINO in current shell: source ${PREFIX}/openvino/setupvars.sh"
    return
  fi

  detect_os
  prepare_common_tools
  prepare_common_dependencies

  log "Detected Ubuntu ${OS_VERSION}."
  run_installation
  rm -rf "${WORK_DIR}"

  log "All OpenVINO installation steps completed successfully."
  log "To load OpenVINO in current shell: source ${INSTALL_ROOT:-/opt/intel}/openvino/setupvars.sh"
}

main "$@"
