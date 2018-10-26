
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNet/Function_Forward.h
/// \brief   OpenNet::Function_Forward

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/Function.h>

namespace OpenNet
{
    class Adapter;

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  The Kernel_Forward class
    /// \endcond
    /// \cond fr
    /// \brief  La classe Kernel_Forward
    /// \endcond
    class Function_Forward : public Function
    {

    public:

        /// \cond en
        /// \brief  Constructor
        /// \endcond
        /// \cond fr
        /// \brief  Constructeur
        /// \endcond
        OPEN_NET_PUBLIC Function_Forward();

        /// \cond en
        /// \brief  Add a destination adapter
        /// \param  aAdapter [---;RW-] The Adapter
        /// \endcond
        /// \cond fr
        /// \brief  Ajouter un Adapter de destination
        /// \param  aAdapter [---;RW-] L'Adapter
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_DESTINATION_ALREADY_SET
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        OPEN_NET_PUBLIC Status AddDestination(Adapter * aAdapter);

        /// \cond en
        /// \brief  Remove a destination adapter
        /// \param  aAdapter [---;RW-] The Adapter
        /// \endcond
        /// \cond fr
        /// \brief  Retirer un Adapter de destination
        /// \param  aAdapter [---;RW-] L'Adapter
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_DESTINATION_NOT_SET
        /// \retval STATUS_NO_DESTINATION_SET
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        OPEN_NET_PUBLIC Status RemoveDestination(Adapter * aAdapter);

        /// \cond en
        /// \brief  Remove all destination Adapter
        /// \endcond
        /// \cond fr
        /// \brief  Retirer tous les Adapter de destination
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NO_DESTINATION_NOT
        OPEN_NET_PUBLIC Status ResetDestinations();

        // ===== Function =======================================================
        virtual Status SetFunctionName(const char * aFunctionName);

        // ===== SourceCode =====================================================

        virtual OPEN_NET_PUBLIC ~Function_Forward();

        virtual Status Display(FILE * aOut) const;

    private:

        Function_Forward(const Function_Forward &);

        const Function_Forward & operator = (const Function_Forward &);

        void GenerateCode();

        uint32_t mDestinations;

    };

}
