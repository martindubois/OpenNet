
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet_Setup/OSDep.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== C ==================================================================
#include <stdio.h>

// ===== Windows ============================================================
#include <Windows.h>

// Functions
/////////////////////////////////////////////////////////////////////////////

void OSDep_ClearScreen()
{
    HANDLE lConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    assert(NULL != lConsole);

    CONSOLE_SCREEN_BUFFER_INFO lInfo;

    BOOL lRetB = GetConsoleScreenBufferInfo(lConsole, &lInfo);
    assert(lRetB);

    COORD lHome      = { 0, 0 };
    DWORD lInfo_char;
    DWORD lSize_char = lInfo.dwSize.X * lInfo.dwSize.Y;

    lRetB = FillConsoleOutputCharacter(lConsole, ' ', lSize_char, lHome, &lInfo_char);
    assert(lRetB                   );
    assert(lSize_char == lInfo_char);

    lRetB = FillConsoleOutputAttribute(lConsole, lInfo.wAttributes, lSize_char, lHome, &lInfo_char);
    assert(lRetB                   );
    assert(lSize_char == lInfo_char);

    lRetB = SetConsoleCursorPosition(lConsole, lHome);
    assert(lRetB);
}

bool OSDep_IsAdministrator()
{
    return true;
}

int OSDep_Reboot()
{
    if (!ExitWindowsEx(EWX_REBOOT, SHTDN_REASON_MAJOR_HARDWARE | SHTDN_REASON_MINOR_INSTALLATION))
    {
        fprintf(stderr, "ERROR  ExitWindowsEx( ,  ) failed\n");
        return __LINE__;
    }

    return 0;
}
