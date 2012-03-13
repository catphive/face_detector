def configuration(parent_package='', top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.misc_util import get_info

    info = get_info('npymath')

    config = Configuration('c_faces',
                            parent_package,
                            top_path)
    config.add_extension('c_faces',
                         ['c_faces.cpp'],
                         extra_info=info,
                         extra_compile_args= ["-Wall", "-Werror", "-O3"])

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)
