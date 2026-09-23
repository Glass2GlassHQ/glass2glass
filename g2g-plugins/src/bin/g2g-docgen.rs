fn main() {
    g2g_plugins::docgen::run(
        &g2g_plugins::registry::default_registry(),
        std::env::args().nth(1),
    );
}
