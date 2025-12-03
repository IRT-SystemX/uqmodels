cd ..
# Test coverage

echo "Running test coverage..."
coverage run -m pytest tests
coverage report > scripts/coverage_report.txt
mkdir -p _static/coverage  
COVERAGE_SCORE=$(grep 'TOTAL' scripts/coverage_report.txt | awk '{print $4}' | sed 's/%//')
echo "Coverage score is $COVERAGE_SCORE"
anybadge --overwrite --label coverage --value=$COVERAGE_SCORE --file=_static/coverage/coverage.svg 50=red 60=orange 75=yellow 100=green
coverage html -d _static/coverage
rm -f scripts/coverage_report.txt