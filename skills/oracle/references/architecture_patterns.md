# Architecture Patterns

## Common Patterns

### Layered Architecture
```
Presentation → Business Logic → Data Access
```
**Use when**: Simple CRUD applications, small teams
**Avoid when**: High performance needed, complex business logic

### Microservices
```
Service A → Service B → Service C
```
**Use when**: Large teams, independent deployment, different scaling needs
**Avoid when**: Small teams, simple apps, real-time requirements

### Event-Driven
```
Producer → Event Bus → Consumer
```
**Use when**: Loose coupling, async processing, audit trails
**Avoid when**: Simple synchronous flows

### Clean Architecture
```
Domain → Use Cases → Interface → Infrastructure
```
**Use when**: Complex business logic, testability priority
**Avoid when**: Simple CRUD apps

## Patterns by Use Case

### State Management
| Pattern | Best For | Avoid When |
|---------|----------|------------|
| Redux | Large apps, time-travel | Small apps |
| Zustand | Simple needs | Complex middleware |
| Recoil | React-specific | Non-React |
| Context | Simple shared state | Frequent updates |

### API Design
| Pattern | Best For | Avoid When |
|---------|----------|------------|
| REST | Standard CRUD | Complex queries |
| GraphQL | Flexible clients | Simple APIs |
| gRPC | High performance | Browser clients |
| WebSocket | Real-time | Simple requests |

### Data Access
| Pattern | Best For | Avoid When |
|---------|----------|------------|
| ORM | Rapid development | High performance |
| Query Builder | SQL control | Complex migrations |
| Raw SQL | Performance | Team SQL skills |

### Caching
| Pattern | Best For | Avoid When |
|---------|----------|------------|
| Redis | Distributed | Single instance |
| Memcached | Simple key-value | Complex data |
| CDN | Static assets | Dynamic content |

## Decision Matrix

### Team Size
- 1-5: Monolith, minimal services
- 5-20: Modular monolith, few services
- 20+: Microservices

### Project Complexity
- Simple: MVC, REST
- Medium: Clean Architecture, GraphQL
- Complex: DDD, Event Sourcing

### Performance Needs
- Low: Standard patterns fine
- High: Caching, async, optimization
- Extreme: Custom solutions

## Anti-Patterns to Avoid

1. **Premature optimization** - Don't over-engineer
2. **Over-abstraction** - YAGNI
3. **Service sprawl** - Don't microservices for everything
4. **Ignorance of constraints** - Consider real limitations
