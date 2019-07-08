
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright &copy; 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNet/Kernel.h
/// \brief      OpenNet::Kernel

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C++ ================================================================
#include <map>

// ===== Includes ===========================================================
#include <OpenNet/SourceCode.h>
#include <OpenNet/StatisticsProvider.h>

// ===== OpenNet ============================================================
class Event;

namespace OpenNet
{

    class UserBuffer;

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  The Kernel class
    /// \endcond
    /// \cond fr
    /// \brief  La classe Kernel
    /// \endcond
    class OPEN_NET_PUBLIC_CLASS Kernel : public SourceCode, public StatisticsProvider
    {

    public:

        /// \cond en
        /// \brief  Constructor
        /// \endcond
        /// \cond fr
        /// \brief  Constructeur
        /// \endcond
        OPEN_NET_PUBLIC Kernel();

        /// \cond en
        /// \brief  Disable OpenCL profiling
        /// \endcond
        /// \cond fr
        /// \brief  D&eacute;sactiver le profiling OpenCL
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_PROFILING_ALREADY_DISABLED
        /// \sa     EnableProfiling, IsProfilingEnabled
        OPEN_NET_PUBLIC Status DisableProfiling();

        /// \cond en
        /// \brief  Enable OpenCL profiling
        /// \endcond
        /// \cond fr
        /// \brief  Activer le profiling OpenCL
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_PROFILING_ALREADY_ENABLED
        /// \sa     DisableProfiling, IsProfilingEnabled
        OPEN_NET_PUBLIC Status EnableProfiling();

        /// \cond en
        /// \brief  Retrieve the build log
        /// \return This method returns the address of an internal buffer.
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir le log de compilation
        /// \retval Cette m&eacute;thode retourne l'adresse d'un espace de
        ///         m&eacute;moire interne.
        /// \endcond
        OPEN_NET_PUBLIC const char * GetBuildLog() const;

        /// \cond en
        /// \brief  Retrieve the code line count
        /// \return This method returns the code line cout.
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir le nombre de lignes de code
        /// \retval Cette m&eacute;thode retourne le nombre de lignes de
        ///         code.
        /// \endcond
        OPEN_NET_PUBLIC unsigned int  GetCodeLineCount();

        /// \cond en
        /// \brief  Retrieve the code lines
        /// \return This method returns the address of an internal buffer.
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir les lignes de code
        /// \retval Cette m&eacute;thode retourne l'adresse d'un espace de
        ///         m&eacute;moire interne.
        /// \endcond
        OPEN_NET_PUBLIC const char ** GetCodeLines();

        /// \cond en
        /// \brief  Retrieve the command queue running this kernel
        /// \retval NULL   The command queue is not assigned
        /// \retval Other  A valid cl_command_queue value
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir la queue de command qui execute ce kernel
        /// \retval NULL   La queue de command n'est pas assign&eacute;
        /// \retval Other  Une valeur valide de cl_command_queue
        /// \endcond
        OPEN_NET_PUBLIC void * GetCommandQueue();

        /// \cond en
        /// \brief  Is the OpenCL profiling enabled?
        /// \note   This method is part of the critical path.
        /// \endcond
        /// \cond fr
        /// \brief  Est-ce que le profiling OpenCL est activ&eacute;?
        /// \note   Cette m&eacute;thode fait partie du chemin critique.
        /// \endcond
        /// \retval false
        /// \retval true
        /// \sa     DisableProfiling, EnableProfiling
        OPEN_NET_PUBLIC bool IsProfilingEnabled() const;

        /// \cond en
        /// \brief  Add a user argument always passed to the kernel
        /// \param  aIndex  The argument index
        /// \param  aArg    The argument
        /// \endcond
        /// \cond fr
        /// \brief  Ajouter un argument utilisateur qui est toujours
        ///         pass&eacutee au kernel
        /// \param  aIndex  L'index de l'argument.
        /// \param  aArg    L'argument
        /// \endcond
        /// \retval false
        /// \retval true
        /// \sa     Processor::AllocateUserBuffer
        OPEN_NET_PUBLIC Status SetStaticUserArgument(unsigned int aIndex, UserBuffer * aArg);

        /// \cond en
        /// \brief  Called to add user arguments to the kernel.
        /// \param  aKernel  The cl_kernel instance
        /// \endcond
        /// \cond fr
        /// \brief  Appel&eacute; pour ajouter des arguments utilisateur au kernel.
        /// \param  aKernel  L'instance de cl_kernel
        /// \endcond
        OPEN_NET_PUBLIC virtual void SetUserKernelArgs(void * aKernel);

        /// \cond en
        /// \brief  Called to add user arguments to the kernel.
        /// \param  aArguments  Argument vector to fill
        /// \endcond
        /// \cond fr
        /// \brief  Appel&eacute; pour ajouter des arguments utilisateur au kernel.
        /// \param  aArguments  Le tableau d'arguments a completer
        /// \endcond
        OPEN_NET_PUBLIC virtual void SetUserKernelArgs(void * * aArguments);

        // ===== SourceCode =================================================
        OPEN_NET_PUBLIC virtual              ~Kernel     ();
        OPEN_NET_PUBLIC virtual Status       AppendCode  (const char * aCode, unsigned int aCodeSize_byte);
        OPEN_NET_PUBLIC virtual Status       ResetCode   ();
        OPEN_NET_PUBLIC virtual Status       SetCode     (const char * aFileName);
        OPEN_NET_PUBLIC virtual Status       SetCode     (const char * aCode, unsigned int aCodeSize_byte);
        OPEN_NET_PUBLIC virtual Status       Display     (FILE * aOut) const;
        OPEN_NET_PUBLIC virtual unsigned int Edit_Remove (const char * aSearch);
        OPEN_NET_PUBLIC virtual unsigned int Edit_Replace(const char * aSearch, const char * aReplace);

        // ===== StatisticsProvider =========================================
        OPEN_NET_PUBLIC virtual Status GetStatistics  (unsigned int * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte, bool aReset);
        OPEN_NET_PUBLIC virtual Status ResetStatistics();

    // Internal:

        void ResetCommandQueue();

        void SetCommandQueue(void * aCommandQueue);

        void AddStatistics( Event * aEvent );

        void * AllocateBuildLog();
        char * AllocateBuildLog( size_t aSize_byte );

    private:

        typedef std::map< unsigned int, UserBuffer * > UserArgumentMap;

        Kernel(const Kernel &);

        const Kernel & operator = (const Kernel &);

        void CodeLines_Count   ();
        void CodeLines_Generate();

        void Invalidate();

        // TODO  OpenNet.Kernel
        //       Normal (Cleanup) - Remove the line buffer for Linux and, if
        //       possible, for Windows.

        char         * mBuildLog         ;
        char         * mCodeLineBuffer   ;
        unsigned int   mCodeLineCount    ;
        const char  ** mCodeLines        ;
        void         * mCommandQueue     ;
        bool           mProfilingEnabled ;
        unsigned int * mStatistics       ;
        uint64_t       mStatisticsSums[3];

        UserArgumentMap mUserArguments;

    };

    // Public
    /////////////////////////////////////////////////////////////////////////

    inline bool Kernel::IsProfilingEnabled() const
    {
        return mProfilingEnabled;
    }

}
