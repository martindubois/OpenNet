
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright &copy; 2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/RegEx.h
/// \brief      RegEx, Reg_Ex_Create, RegEx_End, RegEx_Execute, RegEx_State,
///             REG_EX_, REG_EX_CREATE_, REG_EX_STATE_ (RT)

// CODE REVIEW  2019-07-16  KMS - Martin Dubois, ing.

#pragma once

// Constants
/////////////////////////////////////////////////////////////////////////////

/// \cond    en
/// \defgroup REG_EX_ Regular expression state code
/// \endcond
/// \cond    fr
/// \defgroup REG_EX_ Code d'&eacute;tat pour les expressions regulieres
/// \endcond
/// \{

/// \cond    en
/// A digit
/// \endcond
/// \cond    fr
/// Un chiffre
/// \endcond
/// \sa      REG_EX_STATE_DIGIT
#define REG_EX_DIGIT     (char)(0xef) 

/// \cond    en
/// Not a digit
/// \endcond
/// \cond    fr
/// Pas un chiffre
/// \endcond
/// \sa      REG_EX_STATE_DIGIT_NOT
#define REG_EX_DIGIT_NOT (char)(0xf0)

/// \cond    en
/// A charactere
/// \endcond
/// \cond    fr
/// Un caractere
/// \endcond
/// \sa      REG_EX_STATE_DOT
#define REG_EX_DOT       (char)(0xf1)

/// \cond    en
/// The end of string
/// \endcond
/// \cond    fr
/// La fin de la cha&icirc;ne
/// \endcond
/// \sa      REG_EX_STATE_END
#define REG_EX_END       (char)(0xf2)

/// \cond    en
/// A group
/// \endcond
/// \cond    fr
/// Un groupe
/// \endcond
/// \sa      REG_EX_STATE_GROUP
#define REG_EX_GROUP     (char)(0xf3)

/// \cond    en
/// The end of the regular expression
/// \endcond
/// \cond    fr
/// La fin de l'expresion reguliere
/// \endcond
/// \sa      REG_EX_STATE_OK
#define REG_EX_OK        (char)(0xf4)

/// \cond    en
/// A or
/// \endcond
/// \cond    fr
/// Un ou
/// \endcond
/// \sa      REG_EX_STATE_OR
#define REG_EX_OR        (char)(0xf5)

/// \cond    en
/// The end of a OR operation
/// \endcond
/// \cond    fr
/// La fin d'un op&eacute;ration OR
/// \endcond
/// \sa      REG_EX_STATE_END
#define REG_EX_OR_END    (char)(0xf6)

/// \cond    en
/// A OR operation including simple elements
/// \endcond
/// \cond    fr
/// Une op&eacute;ration OR qui n'inclue que des elements simples
/// \endcond
/// \sa      REG_EX_STATE_FAST
#define REG_EX_OR_FAST   (char)(0xf7)

/// \cond    en
/// A OR NOT
/// \endcond
/// \cond    fr
/// Un OR NOT
/// \endcond
/// \sa      REG_EX_STATE_OR_NOT
#define REG_EX_OR_NOT    (char)(0xf8)

/// \cond    en
/// A charactere range
/// \endcond
/// \cond    fr
/// Un ensemble de caracteres cons&eacute;cutif
/// \endcond
/// \sa      REG_EX_STATE_RANGE
#define REG_EX_RANGE     (char)(0xf9)

/// \cond    en
/// The end of a groupe or OR operation
/// \endcond
/// \cond    fr
/// La fin d'un groupe ou d'une op&eacute;ration OR
/// \endcond
/// \sa      REG_EX_STATE_RETURN
#define REG_EX_RETURN    (char)(0xda)

/// \cond    en
/// A space
/// \endcond
/// \cond    fr
/// Une espace
/// \endcond
/// \sa      REG_EX_STATE_SPACE
#define REG_EX_SPACE     (char)(0xfb)

/// \cond    en
/// Not a space
/// \endcond
/// \cond    fr
/// Pas une espace
/// \endcond
/// \sa      REG_EX_STATE_SPACE_NOT
#define REG_EX_SPACE_NOT (char)(0xfc)

/// \cond    en
/// The begining of the string
/// \endcond
/// \cond    fr
/// Le d&eacute;but de la cha&icirc;ne
/// \endcond
/// \sa      REG_EX_STATE_START
#define REG_EX_START     (char)(0xfd)

/// \cond    en
/// An alpha-numeric charactere
/// \endcond
/// \cond    fr
/// Un caractere alpha-num&eacute;rique
/// \endcond
/// \sa      REG_EX_STATE_WORD
#define REG_EX_WORD      (char)(0xfe)

/// \cond    en
/// An alpha-numeric charactere
/// \endcond
/// \cond    fr
/// Un caractere alpha-num&eacute;rique
/// \endcond
/// \sa      REG_EX_STATE_WORD_NOT
#define REG_EX_WORD_NOT  (char)(0xff)

/// \}

/// \cond    en
/// \defgroup REG_EX_FlagAndLink  Description of the mFlagAndLink field in
///                               the RegEx_State structure
/// \endcond
/// \cond    fr
/// \defgroup REG_EX_FlagAndLink  Description du champ mFlagAndLink de la
///                               structure RegEx_State
/// \endcond
/// \{

/// \cond    en
/// This bit indicates if the state is the beginning of an OR operation
/// \endcond
/// \cond    fr
/// Ce bit indique si un &eacute;tat est le d&eacute;but d'une
/// op&eacute;ration OR
/// \endcond
#define REG_EX_FLAG_OR    0x8000

/// \cond    en
/// These bits indicate the index of the destination state
/// \endcond
/// \cond    fr
/// Ces bits indiquent l'indice de l'&eacute;tat de destination
/// \endcond
#define REG_EX_LINK_MASK  0x7fff

/// \}

// Macros
/////////////////////////////////////////////////////////////////////////////

/// \cond    en
/// \brief   This macro call RegEx_Create to create instance from the state
///          table
/// \endcond
/// \cond    fr
/// \brief   Cette macro appel RegEx_Create pour cr&eacute;er un instance
///          &agrave; partir de la table d'&eacute;tats
/// \endcond
/// \sa      RegEx_Create
#define REG_EX_CREATE(T,S,C)  RegEx_Create( (T), (S), (C), sizeof(S) / sizeof(S[0]) )

/// \cond    en
/// \defgroup REG_EX_STATE  Macros used to create state in the table
/// \endcond
/// \cond    fr
/// \defgroup REG_EX_STATE  Macros utilis&eacute;s pour cr&eacute;er les
///           &eacute;tats dans la table
/// \endcond
/// \{

/// \cond    en
/// A specific charactere
/// \endcond
/// \cond    fr
/// Un caractere specifique
/// \endcond
#define REG_EX_STATE(C,I,A)          { (C)             , (I), (A),   0 }

/// \cond    en
/// A digit
/// \endcond
/// \cond    fr
/// Un chiffre
/// \endcond
/// \sa      REG_EX_DIGIT
#define REG_EX_STATE_DIGIT(I,A)      { REG_EX_DIGIT    , (I), (A),   0 }

/// \cond    en
/// Not a digit
/// \endcond
/// \cond    fr
/// Pas un chiffre
/// \endcond
/// \sa      REG_EX_DIGIT_NOT
#define REG_EX_STATE_DIGIT_NOT(I,A)  { REG_EX_DIGIT_NOT, (I), (A),   0 }

/// \cond    en
/// A charactere
/// \endcond
/// \cond    fr
/// Un caractere
/// \endcond
/// \sa      REG_EX_DOT
#define REG_EX_STATE_DOT(I,A)        { REG_EX_DOT      , (I), (A),   0 }

/// \cond    en
/// The end of string
/// \endcond
/// \cond    fr
/// La fin de la cha&icirc;ne
/// \endcond
/// \sa      REG_EX_END
#define REG_EX_STATE_END             { REG_EX_END      ,   0,   0,   0 }

/// \cond    en
/// A group
/// \endcond
/// \cond    fr
/// Un groupe
/// \endcond
/// \sa      REG_EX_GROUP
#define REG_EX_STATE_GROUP(I,A,L)    { REG_EX_GROUP    , (I), (A), (L) }

/// \cond    en
/// Regular expression end
/// \endcond
/// \cond    fr
/// Fin de l'expression reguliere
/// \endcond
/// \sa      REG_EX_OK
#define REG_EX_STATE_OK              { REG_EX_OK       ,   0,   0,   0 }

/// \cond    en
/// A or
/// \endcond
/// \cond    fr
/// Un ou
/// \endcond
/// \sa      REG_EX_OR
#define REG_EX_STATE_OR(I,A,L)       { REG_EX_OR       , (I), (A), (L) }

/// \cond    en
/// The end of a OR operation
/// \endcond
/// \cond    fr
/// La fin d'un op&eacute;ration OR
/// \endcond
/// \sa      REG_EX_OR_END
#define REG_EX_STATE_OR_END          { REG_EX_OR_END   ,   0,   0,   0 }

/// \cond    en
/// A OR operation including simple elements
/// \endcond
/// \cond    fr
/// Une op&eacute;ration OR qui n'inclue que des elements simples
/// \endcond
/// \sa      REG_EX_OR_FAST
#define REG_EX_STATE_OR_FAST(I,A,L)  { REG_EX_OR_FAST  , (I), (A), (L) }

/// \cond    en
/// A OR NOT
/// \endcond
/// \cond    fr
/// Un OR NOT
/// \endcond
/// \sa      REG_EX_OR_NOT
#define REG_EX_STATE_OR_NOT(I,A,L)   { REG_EX_OR_NOT   , (I), (A), (L) }

/// \cond    en
/// A charactere range
/// \endcond
/// \cond    fr
/// Un ensemble de caracteres cons&eacute;cutif
/// \endcond
/// \sa      REG_EX_RANGE
#define REG_EX_STATE_RANGE(B,E)      { REG_EX_RANGE    , (B), (E),   0 }

/// \cond    en
/// The end of a groupe or OR operation
/// \endcond
/// \cond    fr
/// La fin d'un groupe ou d'une op&eacute;ration OR
/// \endcond
/// \sa      REG_EX_RETURN
#define REG_EX_STATE_RETURN(L)       { REG_EX_RETURN   ,   0,   0, (L) }

/// \cond    en
/// A space
/// \endcond
/// \cond    fr
/// Une espace
/// \endcond
/// \sa      REG_EX_SPACE
#define REG_EX_STATE_SPACE(I,A)      { REG_EX_SPACE    , (I), (A),   0 }

/// \cond    en
/// Not a space
/// \endcond
/// \cond    fr
/// Pas une espace
/// \endcond
/// \sa      REG_EX_SPACE_NOT
#define REG_EX_STATE_SPACE_NOT(I,A)  { REG_EX_SPACE_NOT, (I), (A),   0 }

/// \cond    en
/// The begining of the string
/// \endcond
/// \cond    fr
/// Le d&eacute;but de la cha&icirc;ne
/// \endcond
/// \sa      REG_EX_START
#define REG_EX_STATE_START           { REG_EX_START    ,   0,   0,   0 }

/// \cond    en
/// An alpha-numeric charactere
/// \endcond
/// \cond    fr
/// Un caractere alpha-num&eacute;rique
/// \endcond
/// \sa      REG_EX_WORD
#define REG_EX_STATE_WORD(I,A)       { REG_EX_WORD     , (I), (A),   0 }

/// \cond    en
/// An alpha-numeric charactere
/// \endcond
/// \cond    fr
/// Un caractere alpha-num&eacute;rique
/// \endcond
/// \sa      REG_EX_WORD_NOT
#define REG_EX_STATE_WORD_NOT(I,A)   { REG_EX_WORD_NOT , (I), (A),   0 }

/// \}

// Data types
/////////////////////////////////////////////////////////////////////////////

/// \struct  RegEx_State
/// \cond    en
/// \brief   State of the machine state for a regular expression
/// \endcond
/// \cond    fr
/// \brief   &Eacute;tat de la machine &agrave; &eacute;tats pour une
///          expression reguliere
/// \endcond
typedef struct
{
    char mC;

    unsigned char mMin;
    unsigned char mMax;

    unsigned int mFlagAndLink;
}
RegEx_State;

/// \cond    en
/// \brief   Context of a state machine of a regular expression
/// \endcond
/// \cond    fr
/// \brief   Contexte de la machine &agrave; &eacute;tats pour une expression
///          reguliere
/// \endcond
typedef struct
{

// Internal

    unsigned char * mCounters;

    OPEN_NET_CONSTANT RegEx_State * mStates;
    unsigned short                  mStateCount;

    unsigned short mThreads  [15];
    unsigned char  mThreadCount  ;
    unsigned char  mThreadCurrent;

    unsigned char mRunning;

}
RegEx;

#ifndef _OPEN_NET_NO_FUNCTION_

// Internal
/////////////////////////////////////////////////////////////////////////////

int RegEx_IsCharValid(char aInput)
{
    return (((9 <= aInput) && ( 10 >= aInput))
        ||  (13 == aInput)
        || ((32 <= aInput) && (126 >= aInput)));
}

unsigned short RegEx_StateIndex_Get(RegEx * aThis)
{
    return aThis->mThreads[aThis->mThreadCurrent];
}

void RegEx_StateIndex_Set(RegEx * aThis, unsigned short aState)
{
    aThis->mThreads[aThis->mThreadCurrent] = aState;
}

// --------------------------------------------------------------------------

unsigned short RegEx_Link_Get(RegEx * aThis)
{
    return (aThis->mStates[RegEx_StateIndex_Get(aThis)].mFlagAndLink & REG_EX_LINK_MASK);
}

void RegEx_Thread_Create(RegEx * aThis, unsigned short aState)
{
    aThis->mThreads[aThis->mThreadCount] = aState;
    aThis->mThreadCount++;
}

void RegEx_Thread_Delete(RegEx * aThis)
{
    aThis->mCounters[RegEx_StateIndex_Get(aThis)] = 0;
    aThis->mThreadCount--;

    for (unsigned int i = aThis->mThreadCurrent; i < aThis->mThreadCount; i++)
    {
        aThis->mThreads[i] = aThis->mThreads[i + 1];
    }

    aThis->mThreadCurrent--;
}

// --------------------------------------------------------------------------

void RegEx_Or_Handle(RegEx * aThis)
{
    while (0 != (aThis->mStates[ RegEx_StateIndex_Get( aThis ) ].mFlagAndLink & REG_EX_FLAG_OR))
    {
        RegEx_Thread_Create(aThis, RegEx_StateIndex_Get( aThis ) );
        aThis->mThreads[aThis->mThreadCurrent]++;
    }
}

void RegEx_Reset(RegEx * aThis)
{
    aThis->mThreadCount   = 0;
    aThis->mThreadCurrent = 0;

    for (unsigned int i = 0; i < aThis->mStateCount; i++)
    {
        aThis->mCounters[i] = 0;
    }

    RegEx_Thread_Create(aThis, 0);

    RegEx_Or_Handle(aThis);
}

void RegEx_StateIndex_Next(RegEx * aThis)
{
    aThis->mCounters[RegEx_StateIndex_Get(aThis)] = 0;

    if (0 != (aThis->mStates[ RegEx_StateIndex_Get( aThis ) ].mFlagAndLink & REG_EX_FLAG_OR))
    {
        aThis->mThreads[aThis->mThreadCurrent]++;
    }

    aThis->mThreads[aThis->mThreadCurrent]++;

    RegEx_Or_Handle(aThis);
}

// --------------------------------------------------------------------------

void RegEx_Counter_Inc(RegEx * aThis)
{
    unsigned short lState = RegEx_StateIndex_Get(aThis);

    aThis->mCounters[lState]++;
    if (aThis->mStates[lState].mMax <= aThis->mCounters[lState])
    {
        RegEx_StateIndex_Next(aThis);
    }
}

int RegEx_Repeat_Min(RegEx * aThis)
{
    unsigned short lState = RegEx_StateIndex_Get(aThis);

    if (aThis->mStates[lState].mMin <= aThis->mCounters[lState])
    {
        RegEx_StateIndex_Next(aThis);
        return 1;
    }

    RegEx_Thread_Delete(aThis);
    return 0;
}

void RegEx_Start(RegEx * aThis)
{
    RegEx_Reset(aThis);

    if (REG_EX_START == aThis->mStates[RegEx_StateIndex_Get(aThis)].mC)
    {
        RegEx_StateIndex_Next(aThis);
    }

    aThis->mRunning = 0;
}

// --------------------------------------------------------------------------

void RegEx_OK(RegEx * aThis)
{
    while (0 < aThis->mThreadCount)
    {
        aThis->mThreadCurrent = 0;

        RegEx_Thread_Delete(aThis);
    }

    RegEx_Start(aThis);
}

// ===== RegEx_Execute_... ==================================================

int RegEx_Execute_C(RegEx * aThis, char aInput)
{
    if (aThis->mStates[RegEx_StateIndex_Get(aThis)].mC == aInput)
    {
        RegEx_Counter_Inc(aThis);
        return 0;
    }

    return RegEx_Repeat_Min(aThis);
}

int RegEx_Execute_Digit(RegEx * aThis, char aInput)
{
    if (('0' <= aInput) && ('9' >= aInput))
    {
        RegEx_Counter_Inc(aThis);
        return 0;
    }

    return RegEx_Repeat_Min(aThis);
}

int RegEx_Execute_Digit_Not(RegEx * aThis, char aInput)
{
    if (('0' > aInput) || ('9' < aInput))
    {
        RegEx_Counter_Inc(aThis);
        return 0;
    }

    return RegEx_Repeat_Min(aThis);
}

void RegEx_Execute_Group(RegEx * aThis)
{
    unsigned short lState = RegEx_StateIndex_Get(aThis);

    if (aThis->mStates[lState].mMin <= aThis->mCounters[lState])
    {
        RegEx_Thread_Create(aThis, lState + 1);
    }

    RegEx_StateIndex_Set(aThis, RegEx_Link_Get(aThis));

    RegEx_Or_Handle(aThis);
}

void RegEx_Execute_Or(RegEx * aThis)
{
    unsigned short lState = RegEx_StateIndex_Get(aThis);

    if (aThis->mStates[lState].mMin <= aThis->mCounters[lState])
    {
        RegEx_Thread_Create(aThis, lState + 1);
    }

    unsigned short lLink = RegEx_Link_Get(aThis);

    RegEx_StateIndex_Set(aThis, lLink);

    for (;;)
    {
        lLink += 2;

        if (REG_EX_OR_END == aThis->mStates[lLink].mC)
        {
            break;
        }

        RegEx_Thread_Create(aThis, lLink);
    }
}

int RegEx_Execute_Or_Fast(RegEx * aThis, char aInput)
{
    unsigned short lLink = RegEx_Link_Get(aThis);

    for (;;)
    {
        switch (aThis->mStates[lLink].mC)
        {
        case REG_EX_OR_END:
            return RegEx_Repeat_Min(aThis);

        case REG_EX_RANGE:
            if ((aThis->mStates[lLink].mMin <= aInput) && (aThis->mStates[lLink].mMax >= aInput))
            {
                RegEx_Counter_Inc(aThis);
                return 0;
            }
            break;

        default:
            if (aThis->mStates[lLink].mC == aInput)
            {
                RegEx_Counter_Inc(aThis);
                return 0;
            }
        }

        lLink++;
    }
}

int RegEx_Execute_Or_Not(RegEx * aThis, char aInput)
{
    unsigned short lLink = RegEx_Link_Get(aThis);

    for (;;)
    {
        switch (aThis->mStates[lLink].mC)
        {
        case REG_EX_OR_END:
            RegEx_Counter_Inc(aThis);
            return 0;

        case REG_EX_RANGE:
            if ((aThis->mStates[lLink].mMin <= aInput) && (aThis->mStates[lLink].mMax >= aInput))
            {
                return RegEx_Repeat_Min(aThis);
            }
            break;

        default :
            if (aThis->mStates[lLink].mC == aInput)
            {
                return RegEx_Repeat_Min(aThis);
            }
        }

        lLink++;
    }
}

void RegEx_Execute_Range(RegEx * aThis, char aInput)
{
    unsigned short lState = RegEx_StateIndex_Get(aThis);

    if ((aThis->mStates[lState].mMin <= aInput) && (aThis->mStates[lState].mMax >= aInput))
    {
        RegEx_StateIndex_Next(aThis);
    }
    else
    {
        RegEx_Thread_Delete(aThis);
    }
}

void RegEx_Execute_Return(RegEx * aThis)
{
    unsigned short lLink = RegEx_Link_Get(aThis);

    RegEx_StateIndex_Set(aThis, lLink);

    RegEx_Counter_Inc(aThis);
}

int RegEx_Execute_Space(RegEx * aThis, char aInput)
{
    switch (aInput)
    {
    case ' ' :
    case '\n':
    case '\r':
    case '\t':
        RegEx_Counter_Inc(aThis);
        return 0;
    }

    return RegEx_Repeat_Min(aThis);
}

int RegEx_Execute_Space_Not(RegEx * aThis, char aInput)
{
    switch (aInput)
    {
    case ' ':
    case '\n':
    case '\r':
    case '\t':
        return RegEx_Repeat_Min(aThis);
    }

    RegEx_Counter_Inc(aThis);
    return 0;
}

int RegEx_Execute_Word(RegEx * aThis, char aInput)
{
    if (   (('0' <= aInput) && ('9' >= aInput))
        || (('a' <= aInput) && ('z' >= aInput))
        || (('A' <= aInput) && ('Z' >= aInput))
        || ( '_' == aInput))
    {
        RegEx_Counter_Inc(aThis);
        return 0;
    }

    return RegEx_Repeat_Min(aThis);
}

int RegEx_Execute_Word_Not(RegEx * aThis, char aInput)
{
    if (   (('0' <= aInput) && ('9' >= aInput))
        || (('a' <= aInput) && ('z' >= aInput))
        || (('A' <= aInput) && ('Z' >= aInput))
        || ( '_' == aInput))
    {
        return RegEx_Repeat_Min(aThis);
    }

    RegEx_Counter_Inc(aThis);
    return 0;
}

// Functions
/////////////////////////////////////////////////////////////////////////////

/// \cond    en
/// \brief   Create an engine and start it.
/// \param   aThis      The instance
/// \param   aStates    The state table
/// \param   aCounters  The memory space for the counters
/// \param   aCount     The number of states
/// \endcond
/// \cond    fr
/// \brief   Cr&eacute;er un moteur et le lancer
/// \param   aThis      L'instance
/// \param   aStates    La table d'&eacute;tats
/// \param   aCounters  L'espace m&eacute;moire pour les compteurs
/// \param   aCount     Le nombre d'&eacute;tats dans la table
/// \endcond
/// \sa      REG_EX_CREATE
void RegEx_Create(RegEx * aThis, OPEN_NET_CONSTANT RegEx_State * aStates, unsigned char * aCounters, unsigned int aCount)
{
    aThis->mCounters   = aCounters;
    aThis->mStateCount = aCount   ;
    aThis->mStates     = aStates  ;

    RegEx_Start(aThis);
}

/// \cond    en
/// \brief   Signal the end of a string
/// \param   aThis      The instance
/// \endcond
/// \cond    fr
/// \brief   Signaler la fin d'une cha&icirc;ne
/// \param   aThis      L'instance
/// \endcond
int RegEx_End(RegEx * aThis)
{
    while (0 < aThis->mThreadCount)
    {
        aThis->mThreadCurrent = 0;

        switch (aThis->mStates[RegEx_StateIndex_Get(aThis)].mC)
        {
        case REG_EX_DIGIT_NOT:
        case REG_EX_DOT      :
        case REG_EX_OR_NOT   :
        case REG_EX_SPACE_NOT:
        case REG_EX_WORD_NOT :
            RegEx_Repeat_Min(aThis);
            break;

        case REG_EX_END:
        case REG_EX_OK :
            RegEx_OK(aThis);
            return 1;

        case REG_EX_GROUP : RegEx_Execute_Group (aThis); break;
        case REG_EX_OR    : RegEx_Execute_Or    (aThis); break;
        case REG_EX_RETURN: RegEx_Execute_Return(aThis); break;

        case REG_EX_DIGIT  : RegEx_Execute_Digit  (aThis, REG_EX_END); break;
        case REG_EX_OR_FAST: RegEx_Execute_Or_Fast(aThis, REG_EX_END); break;
        case REG_EX_SPACE  : RegEx_Execute_Space  (aThis, REG_EX_END); break;
        case REG_EX_WORD   : RegEx_Execute_Word   (aThis, REG_EX_END); break;
        default            : RegEx_Execute_C      (aThis, REG_EX_END); break;
        }
    }

    RegEx_Start(aThis);
    return 0;
}

/// \cond    en
/// \brief   Process a charactere
/// \param   aThis   The instance
/// \param   aInput  The charactere
/// \endcond
/// \cond    fr
/// \brief   Traiter un caractere
/// \param   aThis   L'instance
/// \param   aInput  Le caractere
/// \endcond
int RegEx_Execute(RegEx * aThis, char aInput)
{
    if (RegEx_IsCharValid(aInput))
    {
        aThis->mRunning = 1;
        aThis->mThreadCurrent = 0;

        while (aThis->mThreadCurrent < aThis->mThreadCount)
        {
            int lContinue = 1;

            do
            {
                switch (aThis->mStates[RegEx_StateIndex_Get(aThis)].mC)
                {
                case REG_EX_DIGIT    : lContinue = RegEx_Execute_Digit    (aThis, aInput); break;
                case REG_EX_DIGIT_NOT: lContinue = RegEx_Execute_Digit_Not(aThis, aInput); break;
                case REG_EX_OR_FAST  : lContinue = RegEx_Execute_Or_Fast  (aThis, aInput); break;
                case REG_EX_OR_NOT   : lContinue = RegEx_Execute_Or_Not   (aThis, aInput); break;
                case REG_EX_SPACE    : lContinue = RegEx_Execute_Space    (aThis, aInput); break;
                case REG_EX_SPACE_NOT: lContinue = RegEx_Execute_Space_Not(aThis, aInput); break;
                case REG_EX_WORD     : lContinue = RegEx_Execute_Word     (aThis, aInput); break;
                case REG_EX_WORD_NOT : lContinue = RegEx_Execute_Word_Not (aThis, aInput); break;

                case REG_EX_DOT   : lContinue = 0; RegEx_Counter_Inc   (aThis); break;
                case REG_EX_END   : lContinue = 0; RegEx_Thread_Delete (aThis); break;
                case REG_EX_GROUP : lContinue = 1; RegEx_Execute_Group (aThis); break;
                case REG_EX_OR    : lContinue = 1; RegEx_Execute_Or    (aThis); break;
                case REG_EX_RETURN: lContinue = 1; RegEx_Execute_Return(aThis); break;

                case REG_EX_OK: RegEx_OK(aThis); return 1;

                case REG_EX_RANGE: lContinue = 0; RegEx_Execute_Range(aThis, aInput); break;

                default: lContinue = RegEx_Execute_C(aThis, aInput); break;
                }
            }
            while (lContinue);

            aThis->mThreadCurrent++;
        }

        if (0 == aThis->mThreadCount)
        {
            RegEx_Reset(aThis);
        }
    }
    else
    {
        if (aThis->mRunning)
        {
            return RegEx_End(aThis);
        }
    }

    return 0;
}

#endif // ! _OPEN_NET_NO_FUNCTION_
