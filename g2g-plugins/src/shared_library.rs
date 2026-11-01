macro_rules! shared_library {
    (
        $(#[$library_attribute:meta])*
        $visibility:vis struct $library:ident = $file:expr;
        when missing return $missing:expr;
        $(
            $(#[$attribute:meta])*
            $function_visibility:vis fn $name:ident = $symbol:literal
                ($($argument:ident: $argument_type:ty),* $(,)?) -> $return_type:ty;
        )*
    ) => {
        // binaries that link the library directly fail to start on machines without it
        $(#[$library_attribute])*
        $visibility struct $library {
            $( $(#[$attribute])* $name: unsafe extern "C" fn($($argument_type),*) -> $return_type, )*
            _library: libloading::Library,
        }

        impl core::fmt::Debug for $library {
            fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str($file)
            }
        }

        impl $library {
            $visibility fn get() -> Option<&'static Self> {
                static LOADED: std::sync::OnceLock<Option<$library>> = std::sync::OnceLock::new();
                LOADED.get_or_init(Self::open).as_ref()
            }

            fn open() -> Option<Self> {
                // SAFETY: the NVIDIA driver libraries need nothing from the caller when loaded.
                let library = unsafe { libloading::Library::new($file) }.ok()?;
                Some(Self {
                    $(
                        $(#[$attribute])*
                        // SAFETY: the declared signature matches the C prototype of the symbol.
                        $name: unsafe {
                            *library
                                .get::<unsafe extern "C" fn($($argument_type),*) -> $return_type>(
                                    concat!($symbol, "\0").as_bytes(),
                                )
                                .ok()?
                        },
                    )*
                    _library: library,
                })
            }
        }

        $(
            $(#[$attribute])*
            $function_visibility unsafe fn $name($($argument: $argument_type),*) -> $return_type {
                match $library::get() {
                    // SAFETY: the caller upholds the contract of the C function.
                    Some(library) => unsafe { (library.$name)($($argument),*) },
                    None => $missing,
                }
            }
        )*
    };
}

pub(crate) use shared_library;
