#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

[ $# -eq 4 ] || { echo "usage: tools/homebrew-formula.sh <version> <tarball tag> <base url> <tarball dir>" >&2; exit 2; }
version=$1 tag=$2 base_url=$3 tarball_dir=$4

tarball_sha256() {
    shasum -a 256 "$tarball_dir/g2g-$tag-$1.tar.gz" | cut -d' ' -f1
}

cat <<FORMULA
class Glass2glass < Formula
  desc "Multimedia pipeline tools g2g-launch and g2g-inspect"
  homepage "https://github.com/Glass2GlassHQ/glass2glass"
  version "$version"
  license "MPL-2.0"

  depends_on :macos

  on_arm do
    url "$base_url/g2g-$tag-aarch64-apple-darwin.tar.gz"
    sha256 "$(tarball_sha256 aarch64-apple-darwin)"
  end
  on_intel do
    url "$base_url/g2g-$tag-x86_64-apple-darwin.tar.gz"
    sha256 "$(tarball_sha256 x86_64-apple-darwin)"
  end

  def install
    libexec.install Dir["*"]
    bin.install_symlink libexec/"g2g-launch", libexec/"g2g-inspect"
    bash_completion.install_symlink libexec/"g2g.bash" => "g2g-launch"
    bash_completion.install_symlink libexec/"g2g.bash" => "g2g-inspect"
  end

  test do
    system bin/"g2g-inspect", "fakesink"
    system bin/"g2g-launch", "-q", "audiotestsrc", "num-buffers=20", "!", "avenc_aac", "!", "fakesink"
  end
end
FORMULA
