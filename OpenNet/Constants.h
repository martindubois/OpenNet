
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Constants.h

#pragma once

// Constants
/////////////////////////////////////////////////////////////////////////////

#define BUILD_LOG_MAX_SIZE_byte (64*1024)

#ifdef _KMS_LINUX_
    #define DEBUG_LOG_FOLDER "/tmp/OpenNetDebugLog"
#endif

#ifdef _KMS_WINDOWS_
    #define DEBUG_LOG_FOLDER "K:\\Dossiers_Actifs\\OpenNet\\DebugLog"
#endif
