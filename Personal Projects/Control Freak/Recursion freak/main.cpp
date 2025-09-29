//
//  main.cpp
//  Control freak
//
//  Created by Nate Calderon on 6/6/23.
//  Project created Fall SEM 2022 MVCC CSC 240 RECURSION

#include <iostream>
#include <string>
#include <iomanip>
#include <cmath>
using namespace std;

void menu()
{
    cout << "\n\nMaster Menu\n\n";
    cout << "1. Binary to Decimal Conversion\n";
    cout << "2. Decimal to Binary Conversion\n";
    cout << "3. Decimal to other Base\n";
    cout << "4. Factorial\n";
    cout << "5. Generate Fibonacci Sequence\n";
    cout << "6. Find Greatest Common Denominator\n";
    cout << "7. Find Least Common Denominator\n";
    cout << "8. Towers of Hanoi\n";
    cout << "9. Sum Array Elements\n";
    cout << "10. Product of Array Elements\n";
    cout << "11. Rubiks Cube Solver (MULTIPLE FILE PROJECT_INCORPORATION OF VISUALS REQUIRES OPENGL VISUALIZER)\n";
    cout << "12. Scientific Calculator\n";
    cout << "13. UNTITLED PROJECT\n";
    cout << "14. UNTITLED PROJECT\n";
    cout << "15. UNTITLED PROJECT\n";
    cout << "16. UNTITLED PROJECT\n";
    cout << "17. UNTITLED PROJECT\n";
    cout << "18. UNTITLED PROJECT\n";
    cout << "19. UNTITLED PROJECT\n";
    cout << "20. End Program\n";
}
int binToDec(int binary_number)
{
    int dec = 0, i = 0, rem;
    while (binary_number != 0)
    {
        rem = binary_number % 10;
        dec = dec + rem * pow(2, i);
        i++;
        binary_number = binary_number / 10;
    }
    return dec;
}
int decToBin(int decimal_number)
{
    if (decimal_number == 0)
        return 0;
    else
        return (decimal_number % 2 + 10 * decToBin(decimal_number / 2));
}
int decToBase(int decimal_number)
{
    if (decimal_number == 0)
        return 0;
    else
        return (decimal_number);
}
int factorial(int n)
{
    if (n > 1)
        return n * factorial(n - 1);
    else
        return 1;
}
int fib(int x)
{
    if ((x == 1) || (x == 0))
    {
        return (x);
    }
    else
    {
        return(fib(x - 1) + fib(x - 2));
    }
}
int gcd(int n, int m)
{
    if ((n >= m) && ((n % m) == 0))
        return(m);
    else
        return gcd(m, (n % m));
}
int lcd(int n, int m)
{
    if (n == 0 || m == 0)
    {
        return 0;  // LCD of 0 is not defined.
    }
    else
    {
        int gcd_result = gcd(n, m);
        if (gcd_result == 1)
        {
            return n * m;  // If the greatest common denominator (GCD) is 1, then the LCD is the product of n and m.
        }
        else
        {
            return (n * m) / gcd_result;  // Calculate LCD using the GCD.
        }
    }
}
/*int lcd(int n, int m)
{

    if ((n >= m) && ((n % m ) == 0))
        return (m);
    else
      return lcd(m,(n % m));
}*/
int towersOfHanoi(int num, string source, string augment, string destination)
{
    if (num == 1)
    {
        cout << "Move Disk 1 from tower " << source << " to tower " << destination << endl;
        return 0;
    }
    towersOfHanoi(num - 1, source, augment, destination);
    cout << "Move Disk " << num << " from tower " << source << " to tower " << destination << endl;
    towersOfHanoi(num - 1, augment, destination, source);
    return 0;
}
int findSum(int A[], int N)
{
    if (N <= 0)
        return 0;
    return (findSum(A, N - 1) + A[N - 1]);
}
int findProd(int A[], int N)
{
    if (N <= 0)
    return 0;
    return (findProd(A, N - 1) * A[N - 1]);
}
void rubiksCube(char s, string color)
{
    
}
void calculator()
{
    char op;
    float num1, num2, PI;
    float result;
    cout << "\nHow can I Math for you today?: \n";
    cout << "1. Addition\n";
    cout << "2. Subtraction\n";
    cout << "3. Multiplication\n";
    cout << "4. Division\n";
    cout << "5. Power\n";
    cout << "6. Root\n";
    cout << "7. SIN\n";
    cout << "8. COS\n";
    cout << "9. TAN\n";
    cout << "10. ARCSIN\n";
    cout << "11. ARCCOS\n";
    cout << "12. ARCTAN\n";
    cout << "13. LOGARITHMIC FUNCTIONS\n";
    cin >> op;
    PI = 3.14159265;
    switch (op)
    {
        case '1':
            cout << "Enter first Integer: ";
            cin >> num1;
            cout << "Enter second Integer: ";
            cin >> num2;
            result = num1 + num2;
            cout << num1 << " + " << num2 << " = " << result;
            break;
        case '2':
            cout << "Enter first Integer: ";
            cin >> num1;
            cout << "Enter second Integer: ";
            cin >> num2;
            result = num1 - num2;
            cout << num1 << " - " << num2 << " = " << result;
            break;
        case'3':
            cout << "Enter first Integer: ";
            cin >> num1;
            cout << "Enter second Integer: ";
            cin >> num2;
            result = num1 * num2;
            cout << num1 << " * " << num2 << " = " << result;
            break;
        case '4':
            cout << "Enter first Integer: ";
            cin >> num1;
            cout << "Enter second Integer: ";
            cin >> num2;
            result = num1 / num2;
            if (num2 == 0)
            {
                cout << "\nreally...dividing by 0...it's 0 dumbass\n";
                
            }
            else
            {
                cout << num1 << " / " << num2 << " = " << result;
            }
           
            break;
        case '5':
            cout << "Enter base number: ";
            cin >> num1;
            cout << "Enter exponent: ";
            cin >> num2;
            cout << num1 << "raised to the " << num2 << " power is " << pow(num1,num2);
            break;
        case '6':
            cout << "Enter base number: ";
            cin >> num1;
            cout << "Enter exponent: ";
            cin >> num2;
            cout << "The square root of " << num1 << " is "<< sqrt(num1) << endl;
            break;
        case '7':
            cout << "Enter #:";
            cin >> num1;
            cout << "SIN of " << num1 << " is " << sin(num1);
            break;
        case '8':
            cout << "Enter #:";
            cin >> num1;
            cout << "COS of " << num1 << " is " << cos(num1);
            break;
        case '9':
            cout << "Enter #: ";
            cin >> num1;
            cout << "TAN of " << num1 << " is " << tan(num1);
            break;
        case '10':
            cout << "Enter #: ";
            cin >> num1;
            cout << "ARCSIN of " << num1 << "is" << asin(num1) * 180 / PI;
            break;
        case '11':
            cout << "Enter #: ";
            cin >> num1;
            cout << "ARCCOS of " << num1 << "is" << acos(num1) * 180 / PI;
            break;
        case '12':
            cout << "Enter #: ";
            cin >> num1;
            cout << "ARCTAN of " << num1 << "is" << atan(num1) * 180 / PI;
            break;
        case '13':
            cout << "Enter #: ";
            cin >> num1;
            cout << "The Logarithm of " << num1 << " is " << log(num1);
            break;
    }
}
int main()
{
    int ans = 0;
    int dec = 0;
    int bin = 0;
    int decBase = 0;
    int x, i = 0;
    int n;
    int n1, n2, result;
    int num = 0;
    int A[6];
    int N;
    menu();
    cout << "\n\nEnter your Choice: ";
    cin >> ans;
    while (ans != 20)
    {
        switch (ans)
        {
        case 1: cout << "\nEnter a Binary Number: ";
            cin >> bin;
            cout << binToDec(bin) << endl;
            break;
        case 2: cout << "\nEnter a Decimal Number: ";
            cin >> dec;
            cout << decToBin(dec) << endl;
            break;
        case 3: cout << "\nEnter a Decimal Number: ";
            cin >> decBase;
            cout << decToBase(decBase) << endl;
            break;
        case 4: cout << "\nEnter a Positive Integer: ";
            cin >> n;
            cout << "Factorial of " << n << " = " << factorial(n) << endl;
            break;
        case 5: cout << "\nEnter the Number of Terms in the Series: ";
            cin >> x;
            cout << "Fibonnaci Sequence: ";
            while (i < x)
            {
                cout << " " << fib(i);
                i++;
            }
            break;
        case 6: cout << "Input the First Integer Number: ";
            cin >> n1;
            cout << "Input the Second Integer Number: ";
            cin >> n2;
            result = gcd(n1, n2);
            cout << "\nGCD of " << n1 << " and " << n2 << " is " << result;
            break;
            case 7: cout << "Input the First Integer Number: ";
                cin >> n1;
                cout << "Input the Second Integer Number: ";
                cin >> n2;
                result = lcd(n1, n2);
                cout << "\nLCD of " << n1 << " and " << n2 << " is " << result;
                break;
        case 8: cout << "\nEnter the Number of Disks: ";
            cin >> num;
            cout << "\nThe Sequence of Moves :\n ";
            towersOfHanoi(num, "I", "III", "II");
            break;
            case 9: cout << "Populate the Array (Five Numbers)\n";
                cout << "First #: ";
                cin >> A[1];
                cout << "Second #: ";
                cin >> A[2];
                cout << "Third #: ";
                cin >> A[3];
                cout << "Fourth #: ";
                cin >> A[4];
                cout << "Final #: ";
                cin >> A[5];
            N = sizeof(A) / sizeof(A[67]);
            cout << findSum(A, N) << endl;
            break;
        case 10:
            N = sizeof(A) / sizeof(A[0]);
            cout << findProd(A, N) << endl;
            break;
        case 11: cout << "\nUnder Construction\n";
                break;
            case 12: calculator();
                break;
        case 13: cout << "\nUnder Construction\n";
                break;
        case 14: cout << "\nUnder Construction\n";
                break;
        case 15: cout << "\nUnder Construction\n";
                break;
        case 16: cout << "\nUnder Construction\n";
                break;
        case 17: cout << "\nUnder Construction\n";
                break;
        case 18: cout << "\nUnder Construction\n";
                break;
        case 19: cout << "\nUnder Construction\n";
                break;
        case 20: cout << "\nThank You for Using this Program.\nProgram Ending...";
            break;
        }
        if (ans == 20)
            break;
        menu();
        cout << "Enter your Choice: ";
        cin >> ans;
    }
    cout << "Bye!";
    return 0;
}

