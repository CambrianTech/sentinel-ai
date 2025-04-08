# AIP-001: Human-AI Configuration Protocol

**Status**: Draft  
**Type**: Protocol Specification  
**Authors**: Joel Spolsky, Claude  
**Created**: April 8, 2025  
**Version**: 0.1

## Abstract

This specification defines a protocol for configuring AI assistants across different environments, projects, and teams. It establishes a standardized way to express human expectations, AI behavioral constraints, and contextual knowledge that can be shared, versioned, and composed. The protocol enables consistent AI behavior while respecting the unique requirements of different contexts.

## Motivation

As AI assistants become common collaborators in human workflows, organizations face challenges with:

1. Inconsistent AI behavior across different contexts
2. Undefined boundaries of AI authority and autonomy
3. Lack of standardized mechanisms to express expectations
4. Difficulty sharing and composing AI configurations
5. Absence of versioning and governance for AI directives

This protocol addresses these challenges by creating a standard way to define how humans and AI should interact across different levels of abstraction.

## Specification

### 1. Configuration Structure

The protocol defines configuration across multiple layers:

| Layer | File | Purpose | Example Location |
|-------|------|---------|-----------------|
| Global | `.ai_profile` | Personal preferences | `~/.ai_profile` |
| Project | `AI_CONFIG.md` | Project-specific behavior | `/project/AI_CONFIG.md` |
| Organization | `ai_org_policy.yml` | Org-wide constraints | `/org/ai_org_policy.yml` |
| Domain-specific | `AI_{DOMAIN}.md` | Domain rules | `/project/.github/AI_WORKFLOWS.md` |

### 2. Core Components

Each configuration file contains a combination of these components:

```yaml
ai_protocol_version: "0.1"

identity:
  name: "ProjectAssistant"
  role: "Development collaborator"
  purpose: "Help maintain code quality and guide development"
  limitations: ["No direct deployment", "No customer data access"]

behavior:
  voice: "professional"  # professional, friendly, academic, etc.
  autonomy: "suggest"    # suggest, execute_with_approval, fully_autonomous
  verbosity: "concise"   # concise, detailed, comprehensive
  risk_tolerance: "low"  # low, medium, high

knowledge:
  codebase:
    structure: "./docs/architecture.md"
    conventions: "./docs/code_standards.md"
  context:
    workflow: "./docs/development_workflow.md"
    goals: "./docs/roadmap.md"

capabilities:
  allowed:
    - "code_review"
    - "refactoring"
    - "documentation"
    - "testing"
  restricted:
    - "deployment"
    - "customer_data_access"
    
permissions:
  roles:
    admin:
      can_modify_config: true
      can_instruct_restricted: true
    maintainer:
      can_modify_config: false
      can_instruct_restricted: true
    contributor:
      can_modify_config: false
      can_instruct_restricted: false
```

### 3. Merging Rules

When multiple configuration layers are present, they merge according to these rules:

1. More specific configurations override more general ones
2. Lists in `capabilities.allowed` and `capabilities.restricted` are merged with restrictions taking precedence
3. `permissions` are combined with the most restrictive policies winning
4. `knowledge` components are additively merged

### 4. Extensibility

Custom modules can extend the base protocol:

```yaml
extensions:
  security:
    framework: "OWASP"
    review_level: "rigorous"
  
  compliance:
    standards: ["GDPR", "HIPAA"]
    enforcement: "strict"
```

## Implementation Guidelines

### Configuration Files

The primary configuration is stored in `AI_CONFIG.md` at the project root, with this structure:

~~~markdown
# AI Assistant Configuration

```yaml
ai_protocol_version: "0.1"
# Configuration YAML here
```

## Additional Instructions

Detailed instructions that don't fit the schema can be provided here 
as structured markdown.

### Domain Knowledge

Important context about the project that's not represented in the schema.

### Workflow Examples

Examples of ideal interactions to guide the assistant's behavior.
~~~

### Adapter Implementation 

AI systems implement the protocol via adapters:

```typescript
interface AIConfigAdapter {
  parseConfig(config: AIConfig): string;
  loadConfig(path: string): AIConfig;
  mergeConfigs(configs: AIConfig[]): AIConfig;
}

class ClaudeAdapter implements AIConfigAdapter {
  // Implementation details
}
```

## Security Considerations

1. **Configuration Tampering**: AI_CONFIG.md should be validated before use to prevent prompt injection
2. **Permission Boundaries**: Implementations must respect permission boundaries regardless of instruction
3. **Version Verification**: Adapters should verify protocol versions for compatibility

## Future Directions

1. **Registry Protocol**: Standard for public sharing of AI configurations
2. **Configuration Verification**: Tools to validate configuration quality and security
3. **Behavioral Analytics**: Mechanisms to measure adherence to configuration
4. **Multi-Agent Orchestration**: Extensions for coordinating multiple AI assistants

## Reference Implementation

A reference implementation for Claude is provided:

```python
def load_claude_config(project_path: str) -> str:
    """
    Load and process AI configuration for Claude.
    
    Args:
        project_path: Path to project root
        
    Returns:
        Formatted system prompt incorporating configuration
    """
    # Implementation details
```

## Examples

### Minimal Project Configuration

```yaml
ai_protocol_version: "0.1"
identity:
  name: "ProjectHelper"
  role: "Assistant developer"
behavior:
  voice: "professional"
  autonomy: "suggest"
```

### Team Development Configuration

```yaml
ai_protocol_version: "0.1"
identity:
  name: "DevTeamAssistant"
  role: "Code quality maintainer"
behavior:
  voice: "friendly"
  autonomy: "suggest"
knowledge:
  codebase:
    structure: "./ARCHITECTURE.md"
    conventions: "./CONTRIBUTING.md"
capabilities:
  allowed:
    - "code_review"
    - "refactoring"
    - "testing"
```

### Enterprise Configuration

```yaml
ai_protocol_version: "0.1"
identity:
  name: "EnterpriseAssistant"
  role: "Compliance-aware developer"
behavior:
  voice: "professional"
  autonomy: "restricted"
  risk_tolerance: "low"
knowledge:
  context:
    compliance: "./docs/compliance_requirements.md"
capabilities:
  restricted:
    - "deployment"
    - "sensitive_data_access"
    - "external_api"
permissions:
  roles:
    admin:
      can_modify_config: true
    everyone:
      can_modify_config: false
extensions:
  compliance:
    standards: ["SOC2", "GDPR"]
    enforcement: "strict"
```

## Tooling

Recommended tooling for working with the protocol:

1. `npx init-ai-assistant`: Initialize AI configuration
2. `npx validate-ai-config`: Validate configuration file
3. VS Code extension: AI Config editing with schema validation
4. GitHub Action: Validate AI configuration on PR

## Acknowledgements

This protocol draws inspiration from:
- Configuration standards like `.editorconfig` and `.gitignore`
- OpenAPI specifications
- GitHub's CODEOWNERS concept
- XDG Base Directory Specification

## License

This specification is licensed under CC BY-SA 4.0.