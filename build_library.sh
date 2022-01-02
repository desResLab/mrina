python setup.py sdist bdist_wheel
auditwheel repair --plat manylinux_2_17_x86_64 dist/mrina-0.1.1-cp38-cp38-linux_x86_64.whl
mv wheelhouse/* dist
rm dist/*-cp38-cp38-linux_x86_64.whl