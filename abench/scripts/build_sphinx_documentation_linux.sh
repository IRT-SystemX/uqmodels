cd ..
cp -R _static docs/source/
cp -R examples docs/source/

# Delete old abench modules

rm -f docs/source/abench*.rst

# Generate package docstring

sphinx-apidoc -o docs/source abench

# Generate HTML

cd docs
make clean
make html

# Clean temp directories
rm -Rf source/_static
rm -Rf source/examples