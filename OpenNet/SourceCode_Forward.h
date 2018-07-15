
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/SourceCode_Forward.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/Adapter.h>
#include <OpenNet/Status.h>

// Functions
/////////////////////////////////////////////////////////////////////////////

extern OpenNet::Status SourceCode_Forward_AddDestination   (OpenNet::SourceCode * aThis, uint32_t * aDestinations, OpenNet::Adapter * aAdapter);
extern OpenNet::Status SourceCode_Forward_RemoveDestination(OpenNet::SourceCode * aThis, uint32_t * aDestinations, OpenNet::Adapter * aAdapter);
extern OpenNet::Status SourceCode_Forward_ResetDestinations(OpenNet::SourceCode * aThis, uint32_t * aDestinations);

extern void SourceCode_Forward_GenerateCode(OpenNet::SourceCode * aThis, uint32_t aDestinations);
