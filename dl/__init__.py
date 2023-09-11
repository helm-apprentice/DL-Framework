is_simple_core = True

if is_simple_core:
    from dl.core_simple import Variable
    from dl.core_simple import Function
    from dl.core_simple import using_config
    from dl.core_simple import no_grad
    from dl.core_simple import as_array
    from dl.core_simple import as_variable
    from dl.core_simple import setup_variable

else:
    pass

setup_variable()
