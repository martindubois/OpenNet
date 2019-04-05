
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet_Setup/OpenNet_Setup.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== C ==================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ===== Import/Includes ====================================================
#include <KmsTool.h>

// ===== Includes ===========================================================
#include <OpenNet/SetupTool.h>

// ===== Common =============================================================
#include "../Common/Version.h"

// ===== OpenNet_Setup ======================================================
#include "OSDep.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

typedef enum
{
    COMMAND_EXIT  ,
    COMMAND_WIZARD,

    COMMAND_QTY
}
Command;

typedef enum
{
    MODE_INSTALL   ,
    MODE_INTERACTIF,
    MODE_UNINSTALL ,
    MODE_WIZARD    ,
}
Mode;

// Constants
/////////////////////////////////////////////////////////////////////////////

static const char * COMMANDS[COMMAND_QTY] =
{
    "Exit"            ,
    "Start the wizard",
};

#define WIZARD_EXIT (-1)

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

static unsigned int DisplayMenu(OpenNet::SetupTool * aSetupTool);
static void         DisplaySeparator();
static void         DisplayTitle(const char * aTitle, bool aDebug);
static unsigned int ReadAnswer(unsigned int aCount);

// ===== Mode ===============================================================
static int Install   (OpenNet::SetupTool * aSetupTool);
static int Interactif(OpenNet::SetupTool * aSetupTool);
static int Uninstall (OpenNet::SetupTool * aSetupTool);
static int Wizard    (OpenNet::SetupTool * aSetupTool);

// ===== Wizard =============================================================
static int          Wizard_Page(OpenNet::SetupTool * aSetupTool, unsigned int * aPage);
static unsigned int Wizard_Page(const char * aTitle, const char * aText, const char * aButton0, const char * aButton1 = NULL);

// Entry point
/////////////////////////////////////////////////////////////////////////////

int main(int aCount, const char * * aVector)
{
    assert(   0 <  aCount );
    assert(NULL != aVector);

    KMS_TOOL_BANNER("OpenNet", "OpenNet_Tool", VERSION_STR, VERSION_TYPE);

    if (!OSDep_IsAdministrator())
    {
        fprintf(stderr, "USER ERROR  OpenNet_Setup must be run as an administrator\n");
        return __LINE__;
    }

    bool lDebug = false;
    Mode lMode  = MODE_INTERACTIF;

    for (int i = 1; i < aCount; i++)
    {
        if      (0 == strcmp("install"  , aVector[i])) { lMode = MODE_INSTALL  ; }
        else if (0 == strcmp("uninstall", aVector[i])) { lMode = MODE_UNINSTALL; }
        else if (0 == strcmp("wizard"   , aVector[i])) { lMode = MODE_WIZARD   ; }
        else if (0 == strcmp("-d"       , aVector[i])) { lDebug = true; }
        else
        {
            fprintf(stderr, "USER ERROR  %s  is not a valid argument\n", aVector[i]);
            return __LINE__;
        }
    }

    OpenNet::SetupTool * lSetupTool = OpenNet::SetupTool::Create(lDebug);
    assert(NULL != lSetupTool);

    int lResult;

    switch (lMode)
    {
    case MODE_INSTALL   : lResult = Install   (lSetupTool); break;
    case MODE_INTERACTIF: lResult = Interactif(lSetupTool); break;
    case MODE_UNINSTALL : lResult = Uninstall (lSetupTool); break;
    case MODE_WIZARD    : lResult = Wizard    (lSetupTool); break;

    default: assert(false);
    }

    lSetupTool->Delete();

    return lResult;
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

unsigned int DisplayMenu(OpenNet::SetupTool * aSetupTool)
{
    assert(NULL != aSetupTool);

    DisplayTitle("Interactif Menu", aSetupTool->IsDebugEnabled());

    unsigned int i;

    unsigned int lCount = aSetupTool->Interactif_GetCommandCount();
    for (i = 0; i < lCount; i++)
    {
        printf(" %u. %s\n\n", i, aSetupTool->Interactif_GetCommandText(i));
    }

    for (i = 0; i < COMMAND_QTY; i++)
    {
        printf(" %u. %s\n\n", i + lCount, COMMANDS[i]);
    }

    return (lCount + COMMAND_QTY);
}

void DisplayTitle(const char * aTitle, bool aDebug)
{
    assert(NULL != aTitle);

    if (!aDebug)
    {
        OSDep_ClearScreen();
    }

    printf(
        "============================================================\n"
        "%s\n"
        "------------------------------------------------------------\n"
        "\n",
        aTitle);
}

unsigned int ReadAnswer(unsigned int aCount)
{
    assert(1 <= aCount);

    for (;;)
    {
        if (1 >= aCount)
        {
            printf("(0): ");
        }
        else
        {
            printf("(0-%u): ", aCount - 1);
        }

        char lAnswer[128];

        if (NULL != fgets(lAnswer, sizeof(lAnswer), stdin))
        {
            if (0 == strlen(lAnswer))
            {
                printf("Please answer the question\n");
            }
            else
            {
                char * lEnd;

                unsigned int lResult = strtoul(lAnswer, &lEnd, 10);
                if ((lAnswer == lEnd) || (aCount <= lResult))
                {
                    printf("The answer is not valid\n");
                }
                else
                {
                    return lResult;
                }
            }
        }
    }
}

// ===== Mode ===============================================================

int Install(OpenNet::SetupTool * aSetupTool)
{
    assert(NULL != aSetupTool);

    printf("Installation ...\n");

    OpenNet::Status lStatus = aSetupTool->Install();
    assert(OpenNet::STATUS_OK == lStatus);

    return Wizard(aSetupTool);
}

int Interactif(OpenNet::SetupTool * aSetupTool)
{
    assert(NULL != aSetupTool);

    for (;;)
    {
        unsigned int lCount = DisplayMenu(aSetupTool);

        unsigned int lAnswer = ReadAnswer(lCount);

        if (aSetupTool->Interactif_GetCommandCount() > lAnswer)
        {
            OpenNet::Status lStatus = aSetupTool->Interactif_ExecuteCommand(lAnswer);
            switch (lStatus)
            {
            case OpenNet::STATUS_OK: break;

            case OpenNet::STATUS_REBOOT_REQUIRED:
                switch (Wizard_Page("Reboot required", "A reboot is required to complete this operation", "Reboot", "Exit"))
                {
                case 0: return OSDep_Reboot();
                case 1: return 0;

                default: assert(false);
                }
                break;

            default :
                fprintf(stderr, "ERROR  The operation failed\n");
                return 0;
            }
        }
        else
        {
            lAnswer -= aSetupTool->Interactif_GetCommandCount();
            switch (lAnswer)
            {
            case COMMAND_EXIT  : return 0;
            case COMMAND_WIZARD: return Wizard(aSetupTool);
                
            default: assert(false);
            }
        }
    }
}

int Uninstall(OpenNet::SetupTool * aSetupTool)
{
    assert(NULL != aSetupTool);

    OpenNet::Status lStatus = aSetupTool->Uninstall();
    switch (lStatus)
    {
    case OpenNet::STATUS_OK: break;

    case OpenNet::STATUS_REBOOT_REQUIRED :
        Wizard_Page("Reboot required", "A reboot is required to complete the uninstallation.\nPlease reboot the computer after the execution of the uninstallation.", "OK");
        break;

    default: return __LINE__;
    }

    return 0;
}

int Wizard(OpenNet::SetupTool * aSetupTool)
{
    assert(NULL != aSetupTool);

    unsigned int lPage   = 0;
    int          lResult = 0;

    for (;;)
    {
        unsigned int lCount = aSetupTool->Wizard_GetPageCount();
        if (lCount <= lPage)
        {
            break;
        }

        lResult = Wizard_Page(aSetupTool, &lPage);
        if (WIZARD_EXIT == lResult)
        {
            lResult = 0;
            break;
        }

        if (0 != lResult)
        {
            break;
        }
    }

    return lResult;
}

// ===== Wizard =============================================================

int Wizard_Page(OpenNet::SetupTool * aSetupTool, unsigned int * aPage)
{
    assert(NULL != aSetupTool);
    assert(NULL != aPage     );

    DisplayTitle(aSetupTool->Wizard_GetPageTitle(*aPage), aSetupTool->IsDebugEnabled());

    printf("%s\n\n", aSetupTool->Wizard_GetPageText(*aPage));

    unsigned int lCount = aSetupTool->Wizard_GetPageButtonCount(*aPage);
    for (unsigned int i = 0; i < lCount; i++)
    {
        printf(" %u. %s\n", i, aSetupTool->Wizard_GetPageButtonText(*aPage, i));
    }

    printf("\n");

    unsigned int lAnswer = ReadAnswer(lCount);

    OpenNet::Status lStatus = aSetupTool->Wizard_ExecutePage(aPage, lAnswer);
    switch (lStatus)
    {
    case OpenNet::STATUS_OK: break;

    case OpenNet::STATUS_REBOOT_REQUIRED :
        switch (Wizard_Page("Reboot required", "A reboot is required to complete this operation", "Reboot", "Exit"))
        {
        case 0: return OSDep_Reboot();
        case 1: return WIZARD_EXIT;

        default: assert(false);
        }
        break;

    default:
        fprintf(stderr, "The operation failed\n");
        return __LINE__;
    }

    return 0;
}

unsigned int Wizard_Page(const char * aTitle, const char * aText, const char * aButton0, const char * aButton1)
{
    assert(NULL != aTitle  );
    assert(NULL != aText   );
    assert(NULL != aButton0);

    DisplayTitle(aTitle, true);

    printf(
        "%s\n"
        "\n"
        " 0. %s\n",
        aText   ,
        aButton0);

    unsigned int lCount = 1;

    if (NULL != aButton1)
    {
        printf(" 1. %s\n", aButton1);
        lCount++;
    }

    printf("\n");

    return ReadAnswer(lCount);
}
