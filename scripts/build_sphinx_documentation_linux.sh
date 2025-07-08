# Copy dependencies
cd ..
cp -R _static docs/source/
cp -R examples docs/source/

# Delete old uqmodels modules

rm -f docs/source/uqmodels*.rst

# Generate package docstring

sphinx-apidoc -o docs/source uqmodels

# Generate HTML

cd docs
make clean
make html

# Clean temp directories
rm -Rf source/_static
rm -Rf source/examples
