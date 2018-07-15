
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Kernel_Functions.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ++ ===============================================================
#include <string>
#include <vector>

// ===== Includes ===========================================================
#include <OpenNet/Function.h>
#include <OpenNet/Kernel.h>

// Class
/////////////////////////////////////////////////////////////////////////////

class Kernel_Functions : public OpenNet::Kernel
{

public:

    Kernel_Functions();

    void AddDispatchCode(const unsigned int * aBufferQty);
    void AddFunction    (const OpenNet::Function & aFunction);

    // ===== SourceCode =====================================================
    virtual ~Kernel_Functions();

private:

    typedef std::vector< std::string >  String_Vector;

    Kernel_Functions(const Kernel_Functions &);

    const Kernel_Functions & operator = (const Kernel_Functions &);

    void AppendCode(const char * aCode);

    String_Vector mFunctionNames;

};
