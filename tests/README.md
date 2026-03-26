# Testing Strategy for Weather Prediction System

## Overview
This directory contains a comprehensive testing suite for the weather prediction system, following industry-standard testing practices and the testing pyramid approach.

## Testing Architecture

### Testing Pyramid
```
    /\
   /  \     E2E Tests (few, slow, expensive)
  /____\    
 /      \   Integration Tests (some, medium speed)
/________\  
Unit Tests (many, fast, cheap)
```

## Directory Structure
```
tests/
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_config.py      # Configuration module tests
│   ├── test_meteorological_processor.py  # Data processor tests
│   ├── test_preprocessing_service.py     # Service layer tests
│   └── test_meteorological_plotter.py    # Visualization tests
├── integration/            # Integration tests (medium speed)
├── e2e/                   # End-to-end tests (slow, comprehensive)
├── conftest.py            # Shared fixtures and configuration
├── pytest.ini            # Pytest configuration
└── run_tests.py          # Test runner script
```

## Current Test Status

### ✅ Unit Tests (8 tests passing)
- **Configuration Module**: 3 tests covering `get_config_for_variable` for temperature, precipitation, and humidity
- **Meteorological Data Processor**: 3 tests covering initialization, data cleaning, and data saving
- **Preprocessing Service**: 2 tests covering initialization and data processing pipeline
- **Meteorological Plotter**: 1 test covering initialization (plotting methods to be implemented)

### Coverage Report
- **Total Coverage**: 46%
- **Well Covered Modules**:
  - `preprocessing_service.py`: 86% coverage
  - `base_processor.py`: 75% coverage
  - `meteorological_processor.py`: 66% coverage
- **Needs Improvement**:
  - `settings.py`: 38% coverage (plot configuration functions not tested)
  - `base_plotter.py`: 28% coverage (plotting methods not implemented)
  - `meteorological_plotter.py`: 11% coverage (plotting methods not implemented)

## Running Tests

### Basic Test Execution
```bash
# Run all tests
python -m pytest tests/

# Run only unit tests
python -m pytest tests/unit/

# Run with verbose output
python -m pytest tests/unit/ -v

# Run with coverage
python -m pytest tests/unit/ --cov=src --cov-report=html
```

### Test Runner Script
```bash
# Run all tests with coverage and reports
python tests/run_tests.py
```

## Test Categories

### Unit Tests
- **Purpose**: Test individual functions and classes in isolation
- **Speed**: Fast (< 1 second per test)
- **Scope**: Single function/class
- **Dependencies**: Mocked external dependencies

### Integration Tests (Planned)
- **Purpose**: Test interactions between components
- **Speed**: Medium (1-10 seconds per test)
- **Scope**: Multiple components working together
- **Dependencies**: Real database/filesystem

### End-to-End Tests (Planned)
- **Purpose**: Test complete workflows
- **Speed**: Slow (10+ seconds per test)
- **Scope**: Full system
- **Dependencies**: Complete environment

## Test Data Management

### Fixtures
- **temp_csv_file**: Temporary CSV file with sample meteorological data
- **meteorological_processor**: Pre-configured data processor instance
- **preprocessing_service**: Pre-configured service instance
- **meteorological_plotter**: Pre-configured plotter instance
- **temp_output_dir**: Temporary directory for test outputs

### Test Data Characteristics
- **Size**: Small, representative datasets
- **Content**: Realistic meteorological data patterns
- **Cleanup**: Automatic cleanup after tests
- **Isolation**: Each test uses fresh data

## Best Practices

### Test Design
1. **Arrange-Act-Assert**: Clear test structure
2. **Descriptive Names**: Test names explain what is being tested
3. **Single Responsibility**: Each test verifies one behavior
4. **Independence**: Tests don't depend on each other

### Code Quality
1. **Mock External Dependencies**: Use mocks for databases, APIs, files
2. **Fast Execution**: Unit tests should run quickly
3. **Clear Assertions**: Use specific assertions
4. **Error Handling**: Test both success and failure cases

### Maintenance
1. **Keep Tests Updated**: Update tests when code changes
2. **Monitor Coverage**: Aim for >80% coverage on critical paths
3. **Review Test Quality**: Regular review of test effectiveness
4. **Documentation**: Keep test documentation current

## Future Enhancements

### Planned Test Additions
1. **Integration Tests**: Test component interactions
2. **E2E Tests**: Test complete workflows
3. **Performance Tests**: Test system performance under load
4. **Security Tests**: Test data validation and security measures

### Coverage Improvements
1. **Plot Configuration Tests**: Test `get_plot_config` functions
2. **Plotting Method Tests**: Test visualization functionality
3. **Error Handling Tests**: Test edge cases and error conditions
4. **Data Validation Tests**: Test data quality checks

## Continuous Integration

### GitHub Actions (Planned)
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.12
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python -m pytest tests/ --cov=src
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure `src` is in Python path
2. **Fixture Errors**: Check fixture definitions in `conftest.py`
3. **Coverage Issues**: Verify source files are included in coverage
4. **Performance Issues**: Use `pytest-xdist` for parallel execution

### Debug Mode
```bash
# Run with debug output
python -m pytest tests/unit/ -v -s --tb=long

# Run specific test
python -m pytest tests/unit/test_config.py::TestConfiguration::test_get_config_for_variable_temperature -v -s
```

## Contributing

### Adding New Tests
1. Follow existing naming conventions
2. Use appropriate fixtures
3. Add to correct test category
4. Update documentation
5. Ensure tests pass before committing

### Test Review Checklist
- [ ] Tests are focused and specific
- [ ] Tests use appropriate mocks
- [ ] Tests handle edge cases
- [ ] Tests are well documented
- [ ] Tests follow naming conventions
- [ ] Tests don't have side effects 