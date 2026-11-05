// Get DOM elements
const displayExpression = document.querySelector('.expression');
const displayResult = document.querySelector('.result');
const buttons = document.querySelectorAll('.btn');

let currentInput = '0';
let expression = '';
let resetDisplay = false;

// Update display
function updateDisplay() {
    displayExpression.textContent = expression;
    displayResult.textContent = currentInput;
}

// Handle number input
function inputNumber(number) {
    if (currentInput === '0' || resetDisplay) {
        currentInput = number;
        resetDisplay = false;
    } else {
        currentInput += number;
    }
    updateDisplay();
}

// Handle decimal point
function inputDecimal() {
    if (resetDisplay) {
        currentInput = '0.';
        resetDisplay = false;
    } else if (!currentInput.includes('.')) {
        currentInput += '.';
    }
    updateDisplay();
}

// Handle operators
function inputOperator(operator) {
    if (expression === '' && operator === '-') {
        // Allow negative numbers
        if (currentInput === '0') {
            currentInput = '-';
            updateDisplay();
            return;
        }
    }
    
    // If last character in expression is an operator, replace it
    if (expression !== '' && ['+', '-', '*', '/'].includes(expression.slice(-1))) {
        expression = expression.slice(0, -1) + operator;
    } else {
        // Add current input and operator to expression
        expression += currentInput + operator;
    }
    
    resetDisplay = true;
    updateDisplay();
}

// Calculate result
function calculate() {
    if (expression === '') return;
    
    try {
        // Combine expression and current input for evaluation
        const evalString = expression + currentInput;
        
        // Validate expression to prevent dangerous eval usage
        if (/^[0-9+\-*/.() ]+$/.test(evalString)) {
            // Use Function constructor for safer evaluation
            const result = new Function('return ' + evalString)();
            
            // Check for valid number result
            if (typeof result === 'number' && isFinite(result)) {
                currentInput = parseFloat(result.toFixed(10)).toString();
            } else {
                currentInput = 'Error';
            }
            
            expression = '';
            resetDisplay = true;
            updateDisplay();
        } else {
            throw new Error('Invalid expression');
        }
    } catch (error) {
        currentInput = 'Error';
        expression = '';
        resetDisplay = true;
        updateDisplay();
    }
}

// Clear everything
function clearAll() {
    currentInput = '0';
    expression = '';
    resetDisplay = false;
    updateDisplay();
}

// Backspace function
function backspace() {
    if (resetDisplay || currentInput.length === 1 || 
        (currentInput.length === 2 && currentInput.startsWith('-'))) {
        currentInput = '0';
        resetDisplay = false;
    } else {
        currentInput = currentInput.slice(0, -1);
    }
    updateDisplay();
}

// Event listeners for buttons
buttons.forEach(button => {
    button.addEventListener('click', () => {
        const value = button.dataset.value;
        
        switch(value) {
            case 'C':
                clearAll();
                break;
            case 'backspace':
                backspace();
                break;
            case '=':
                calculate();
                break;
            case '+':
            case '-':
            case '*':
            case '/':
                inputOperator(value);
                break;
            case '.':
                inputDecimal();
                break;
            default:
                // Handle number buttons
                inputNumber(value);
        }
    });
});

// Keyboard support
document.addEventListener('keydown', (event) => {
    const key = event.key;
    
    // Prevent default behavior for keys we're handling
    if ('0123456789+-*/.=Enter'.includes(key) || key === 'Backspace' || key === 'Escape') {
        event.preventDefault();
    }
    
    switch(key) {
        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
            inputNumber(key);
            break;
        case '.':
            inputDecimal();
            break;
        case '+':
        case '-':
        case '*':
        case '/':
            inputOperator(key);
            break;
        case 'Enter':
        case '=':
            calculate();
            break;
        case 'Backspace':
            backspace();
            break;
        case 'Escape':
            clearAll();
            break;
    }
});

// Initialize display
updateDisplay();