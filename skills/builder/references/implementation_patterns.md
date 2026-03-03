# Implementation Patterns

## Design Patterns

### Singleton
```typescript
class Database {
  private static instance: Database;
  
  private constructor() {}
  
  static getInstance(): Database {
    if (!Database.instance) {
      Database.instance = new Database();
    }
    return Database.instance;
  }
}
```

### Factory
```typescript
interface Handler {
  handle(data: unknown): Promise<void>;
}

class HandlerFactory {
  static create(type: string): Handler {
    switch (type) {
      case 'api': return new ApiHandler();
      case 'file': return new FileHandler();
      default: throw new Error(`Unknown handler: ${type}`);
    }
  }
}
```

### Repository
```typescript
interface UserRepository {
  findById(id: string): Promise<User | null>;
  findAll(): Promise<User[]>;
  save(user: User): Promise<void>;
  delete(id: string): Promise<void>;
}
```

## Error Handling Patterns

### Result Type
```typescript
type Result<T, E = Error> = 
  | { ok: true; value: T }
  | { ok: false; error: E };

async function getUser(id: string): Promise<Result<User>> {
  try {
    const user = await db.findUser(id);
    return { ok: true, value: user };
  } catch (e) {
    return { ok: false, error: e as Error };
  }
}
```

### Error Boundary (React)
```typescript
class ErrorBoundary extends React.Component {
  state = { hasError: false };
  
  static getDerivedStateFromError() {
    return { hasError: true };
  }
  
  render() {
    if (this.state.hasError) {
      return <FallbackComponent />;
    }
    return this.props.children;
  }
}
```

## Validation Patterns

### Zod Schema
```typescript
const UserSchema = z.object({
  email: z.string().email(),
  name: z.string().min(1).max(100),
  age: z.number().int().min(0).optional(),
});

type User = z.infer<typeof UserSchema>;
```

## Testing Patterns

### Arrange-Act-Assert
```typescript
describe('Calculator', () => {
  it('adds two numbers', () => {
    // Arrange
    const calc = new Calculator();
    
    // Act
    const result = calc.add(2, 3);
    
    // Assert
    expect(result).toBe(5);
  });
});
```

### Mock Pattern
```typescript
const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

mockFetch.mockResolvedValue({
  json: async () => ({ data: 'test' })
});
```

## Performance Patterns

### Memoization
```typescript
const expensive = useMemo(() => {
  return computeExpensiveValue(a, b);
}, [a, b]);
```

### Virtualization
```typescript
const VirtualList = ({ items }) => (
  <List>
    {items.slice(startIndex, endIndex).map(item => (
      <ListItem key={item.id} item={item} />
    ))}
  </List>
);
```

## Code Organization

### Feature-Based
```
src/
├── features/
│   ├── auth/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── api/
│   │   └── index.ts
│   └── users/
│       └── ...
├── shared/
│   ├── components/
│   └── utils/
└── app/
```

### Clean Architecture
```
src/
├── domain/        # Business entities
├── application/   # Use cases
├── infrastructure/# External services
└── presentation/ # UI
```
