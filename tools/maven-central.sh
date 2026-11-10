#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/kotlin-tools.sh"

GROUP_ID=io.github.glass2glasshq
ARTIFACT_ID=g2g-android
VERSION="$(sed -n 's/^version = "\(.*\)"$/\1/p' "$REPO/Cargo.toml" | head -n 1)"

write_pom() {
    cat <<POM
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>$GROUP_ID</groupId>
  <artifactId>$ARTIFACT_ID</artifactId>
  <version>$VERSION</version>
  <packaging>aar</packaging>
  <name>glass2glass for Android</name>
  <description>Kotlin bindings and native libraries for glass2glass multimedia pipelines.</description>
  <url>https://github.com/Glass2GlassHQ/glass2glass</url>
  <licenses>
    <license>
      <name>MPL-2.0</name>
      <url>https://www.mozilla.org/en-US/MPL/2.0/</url>
    </license>
  </licenses>
  <developers>
    <developer>
      <name>Aaron Boxer</name>
      <email>boxerab@protonmail.com</email>
    </developer>
  </developers>
  <scm>
    <connection>scm:git:https://github.com/Glass2GlassHQ/glass2glass.git</connection>
    <developerConnection>scm:git:ssh://git@github.com/Glass2GlassHQ/glass2glass.git</developerConnection>
    <url>https://github.com/Glass2GlassHQ/glass2glass</url>
  </scm>
  <dependencies>
    <dependency>
      <groupId>net.java.dev.jna</groupId>
      <artifactId>jna</artifactId>
      <version>$JNA_VERSION</version>
      <type>aar</type>
      <scope>runtime</scope>
    </dependency>
  </dependencies>
</project>
POM
}

cmd_layout() {
    local aar generated layout version_dir stem scratch
    aar=$(realpath "$1") generated=$(realpath "$2") layout=$3
    version_dir="$layout/${GROUP_ID//.//}/$ARTIFACT_ID/$VERSION"
    mkdir -p "$version_dir"
    version_dir=$(realpath "$version_dir")
    stem="$version_dir/$ARTIFACT_ID-$VERSION"
    cp "$aar" "$stem.aar"
    write_pom > "$stem.pom"
    (cd "$generated" && zip -qr "$stem-sources.jar" .)
    (cd "$REPO/g2g-uniffi/android" && zip -q "$stem-sources.jar" Surfaces.kt)
    # maven central accepts a placeholder javadoc jar
    scratch=$(mktemp -d)
    echo "The API documentation is in the sources jar and g2g-uniffi/README.md." > "$scratch/README.txt"
    (cd "$scratch" && zip -q "$stem-javadoc.jar" README.txt)
    rm -r "$scratch"
}

cmd_bundle() {
    local layout=$1 bundle file
    bundle=$(realpath -m "$2")
    while read -r file; do
        gpg --batch --yes --armor --detach-sign "$file"
        md5sum "$file" | cut -d' ' -f1 > "$file.md5"
        sha1sum "$file" | cut -d' ' -f1 > "$file.sha1"
    done < <(find "$layout" -type f)
    (cd "$layout" && zip -qr "$bundle" .)
}

case "${1:-}" in
layout) cmd_layout "$2" "$3" "$4" ;;
bundle) cmd_bundle "$2" "$3" ;;
*)
    echo "usage: tools/maven-central.sh layout <aar> <generated kotlin dir> <layout dir> | bundle <layout dir> <zip>" >&2
    exit 2
    ;;
esac
