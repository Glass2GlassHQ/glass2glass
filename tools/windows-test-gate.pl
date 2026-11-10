#!/usr/bin/perl
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# usage: tools/windows-test-gate.pl "<enabled features>" <test.rs>...
# Prints "<stem> run" or "<stem> skip cfg(...)" for each integration test file,
# evaluating its crate-level #![cfg(...)] gates for x86_64-pc-windows-msvc with
# those g2g-plugins features on. tools/windows-test-suite.sh uses it to skip,
# without building, the test files that would compile to an empty binary.
use strict;
use warnings;

my %on = map { $_ => 1 } split ' ', shift @ARGV;
my %target = (
    target_os => 'windows',
    target_family => 'windows',
    target_env => 'msvc',
    target_arch => 'x86_64',
    target_pointer_width => '64',
    target_vendor => 'pc',
);

# Evaluate one cfg predicate, consuming it from the front of $$s.
sub ev {
    my ($s) = @_;
    $$s =~ s/^\s+//;
    if ($$s =~ s/^(all|any|not)\s*\(//) {
        my $op = $1;
        my @v;
        while (1) {
            $$s =~ s/^\s+//;
            last if $$s =~ s/^\)//;
            push @v, ev($s);
            $$s =~ s/^\s*,//;
        }
        return $op eq 'all' ? !grep { !$_ } @v
             : $op eq 'any' ? !!grep { $_ } @v
             : !$v[0];
    }
    if ($$s =~ s/^(\w+)\s*=\s*"([^"]*)"//) {
        return $1 eq 'feature' ? !!$on{$2} : (($target{$1} // '') eq $2);
    }
    if ($$s =~ s/^(\w+)//) {
        # bare names: `windows` and `test` hold here; `unix` and the rest do not
        return $1 eq 'windows' || $1 eq 'test';
    }
    die "cannot parse cfg at: $$s\n";
}

for my $file (@ARGV) {
    open my $fh, '<', $file or die "$file: $!";
    my $src = do { local $/; <$fh> };
    (my $stem = $file) =~ s{.*[/\\]}{};
    $stem =~ s/\.rs$//;
    my $verdict = 'run';
    # every crate-level gate must hold
    while ($src =~ /^#!\[cfg\((.*?)\)\]\s*$/msg) {
        my $expr = $1;
        (my $shown = $expr) =~ s/\s+/ /g;
        if (!ev(\$expr)) {
            $verdict = "skip cfg($shown)";
            last;
        }
    }
    print "$stem $verdict\n";
}
