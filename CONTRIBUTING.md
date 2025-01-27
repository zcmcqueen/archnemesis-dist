# Contribution Guidelines for archNEMESIS

Thank you for considering contributing to archNEMESIS! By participating, you’re helping to make this project even better. To ensure a smooth collaboration, please follow these guidelines.

## 1. Getting started

- Familiarise yourself with the project by reviewing the README and existing documentation.
- Join our [Discord](https://discord.gg/Te43qbrVFK) to engage with the NEMESIS community and ask for advice on how to use the code.

## 2. Reporting issues

- Report any issues through the GitHub issue tracker.
- Provide a detailed description of the issue, including:
    - Use clear and descriptive titles.
    - Steps to reproduce the problem
    - Expected vs. actual behaviour
    - Relevant logs or screenshots, if applicable

## 3. Submitting Pull Requests

- Fork the repository and create a new branch with a meaningful name.
- Stick to the project's coding style and naming conventions.
- Avoid divergence with original NEMESIS code:
    - When introducing new parametersiations, check the documentation of NEMESIS to see if the numerical identifier is already taken by some other model in the Fortran version. This way we will ensure backward compatibility between the two codes.
- Comment the code adequately:
    - When introducing methods or functions, include headers.
    - When introducing new attributes in the classes, include what they are and their type and expected dimensions in the header of the class.
- Run all existing tests and ensure they pass before submitting your pull request.
- Include tests for your changes whenever possible.
- Remove commented-out code, unnecessary console logs, or debugging code before submitting.

## 4. Review process

- Be open to feedback during the review process.
- Address requested changes promptly to keep the review process moving.
- Engage with maintainers or other contributors to clarify any concerns.

## 5. Update documentation

- Update documentation as necessary to reflect your changes.
- If you are adding a new feature, consider including a jupyter notebook example in the documentation.

## 6. Community Support

If you need help, feel free to open a discussion thread or ask questions in the community [Discord](https://discord.gg/Te43qbrVFK) channel.

## 7. License

By contributing, you agree that your contributions will be licensed under the same open-source [GNU General Public License v3](LICENSE) as this project.