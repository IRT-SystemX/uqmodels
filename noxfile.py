import nox
from nox import Session, options
from nox_uv import session

options.default_venv_backend = "uv"


@nox.session
def test(s: nox.Session) -> None:
    """Run full test suite with coverage reporting."""
    s.run(
        "uv",
        "pip",
        "install",
        ".",
        "-r",
        "requirements.txt",
        "-r",
        "requirements_test.txt",
        external=True,
    )
    s.run(
        "pytest",
        "tests",
        "--cov=uqmodels",
        "--cov-report=term",
        "--cov-report=html:logs/htmlcov",
        "--cov-report=json:logs/coverage.json",
        *s.posargs,
    )


@session(python=["3.11", "3.12", "3.13"])
def compatibility(s: Session) -> None:
    s.run(
        "uv",
        "pip",
        "install",
        "-r",
        "requirements.txt",
        "-r",
        "requirements_test.txt",
        external=True,
    )
    s.run("pytest", "tests", *s.posargs)


@nox.session
def test_complexity(s: nox.Session) -> None:
    """Check cyclomatic complexity of test code (tests/). Fails if any function exceeds 15."""
    s.run("uv", "pip", "install", "complexipy", external=True)
    s.run("complexipy", "tests", "--max-complexity-allowed", "15")
