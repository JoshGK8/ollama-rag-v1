# Multi-Interface Detection & Clarification System

## Problem Solved
When users ask "how to" do something, they often get CLI instructions when they wanted GUI steps, or vice versa. The system now intelligently detects when multiple interfaces are available and asks users to clarify their preference.

## Key Features

### 1. **Automatic Interface Detection** 🔍
- Scans retrieved documents for interface-specific keywords
- Detects: GUI, CLI, API, Mobile App, Desktop App
- Works generically for any company's documentation

### 2. **Smart Question Classification** 🧠
- Identifies "how to" questions that might have multiple solutions
- Patterns: "how to", "how do i", "steps to", "configure", "setup", etc.
- Only triggers for procedural questions, not informational ones

### 3. **User-Friendly Clarification** 💬
- Presents clear interface options with descriptions
- Example: "GUI (web dashboard/graphical interface)"
- Allows users to specify their preferred interface

### 4. **Interface-Specific Filtering** 🎯
- Re-queries with interface preference
- Prioritizes documents containing the preferred interface terms
- Provides focused, relevant instructions

## Generic Implementation

### Interface Patterns (works for any company):
```python
interface_patterns = {
    'gui': ['web interface', 'dashboard', 'web ui', 'browser', 'click', 'button'],
    'cli': ['command line', 'terminal', 'cli', 'command', 'script', 'shell'],
    'api': ['api', 'endpoint', 'rest', 'http', 'post', 'get', 'json'],
    'mobile': ['mobile app', 'smartphone', 'android', 'ios', 'app'],
    'desktop': ['desktop app', 'application', 'software', 'program']
}
```

## User Experience Flow

### Before Enhancement:
```
User: "How do I create a wallet?"
System: [Returns CLI instructions mixed with GUI info - confusing]
```

### After Enhancement:
```
User: "How do I create a wallet?"
System: "I found instructions for multiple interfaces to create a wallet.

Available interfaces: 
- GUI (web dashboard/graphical interface)
- CLI (command line interface) 
- API (REST API/programmatic interface)

Which interface would you prefer to use?"

User: "gui"
System: [Returns focused GUI-specific instructions]
```

## Test Results

✅ **Interface Detection Working:**
- Correctly detects 5 interfaces: API, CLI, Desktop, GUI, Mobile
- Triggers clarification for "how to" questions
- Skips clarification for informational questions

✅ **Focused Answers:**
- API preference → Returns JSON examples and endpoint documentation
- CLI preference → Returns command-line specific instructions  
- GUI preference → Returns web interface navigation steps

✅ **Generic Implementation:**
- Works with any company's documentation structure
- No hardcoded GK8-specific terms
- Easily adaptable to other domains

## Technical Implementation

### New Methods Added:
- `_is_how_to_question()` - Detects procedural questions
- `_detect_multiple_interfaces()` - Scans for interface types
- `_generate_interface_clarification()` - Creates user prompt
- `_filter_docs_by_interface()` - Filters by preference

### Enhanced Query Method:
- Added `interface_preference` parameter
- Returns `needs_clarification` flag
- Includes `interfaces_available` list

### CLI Integration:
- Interactive interface selection
- Graceful handling of user choices
- Rich formatting for better UX

## Business Value

### For Users:
- **Reduced Friction**: Get the right instructions for their preferred interface
- **Better Accuracy**: No more confusion between CLI and GUI steps
- **Time Savings**: Direct path to relevant information

### For Enterprises:
- **Improved Support**: Fewer support tickets from confused users
- **Professional UX**: Intelligent system that understands user context
- **Scalable**: Works automatically as documentation grows

### For Documentation Teams:
- **No Extra Work**: System automatically detects interfaces from existing docs
- **Better Metrics**: Can track interface preferences
- **Future-Proof**: Easily supports new interfaces as they're added

## Usage Examples

### In Code:
```python
# Automatic detection
result = rag.query("How do I setup authentication?")
if result.get('needs_clarification'):
    # Handle interface selection
    
# Direct interface specification  
result = rag.query("How do I setup auth?", interface_preference="gui")
```

### In CLI:
```bash
$ python src/rag_cli.py
> How do I create a policy?

# System automatically detects multiple interfaces and prompts user
# User selects preference and gets focused answer
```

## Future Enhancements

1. **Remember Preferences**: Store user's preferred interface
2. **Confidence Scoring**: Weight by interface relevance  
3. **Hybrid Answers**: Show multiple interfaces when beneficial
4. **Interface Popularity**: Track most-used interfaces

## Result

A production-ready RAG system that intelligently handles multi-interface documentation, providing users with exactly the type of instructions they need without confusion or irrelevant information.