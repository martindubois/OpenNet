
// Author   KMA - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/Base.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/System.h>

// Class
/////////////////////////////////////////////////////////////////////////////

class Base
{

public:

    Base();

    ~Base();

    int Init();

    OpenNet::System * mSystem;

};
