import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
from typing import List

pd.options.display.float_format = '{:20,.2f}'.format


#Define Mortgage Class
class Mortgage:
    #This class is initialized with outstanding principal left on the mortgage,
    #the coupon rate of the mortgage, and the number of months left (term).
    def __init__(self, outstanding_principal, coupon_rate, term):
        self.outstanding_principal = outstanding_principal
        self.coupon_rate = coupon_rate        
        self.term = int(term)
        self.payment = self.amortization() #We use an amortization function to define the monthly payments
        self.cashflow_df = self.cashflows() #We convert the mortgages to a dataframe with cashflows
    
    
    #Amortization function uses the monthly coupon rate, term number, and outstanding
    #principal to calculate the monthly payments required
    def amortization(self):
        x = (1 + (self.coupon_rate/12)) ** self.term
        return self.outstanding_principal * (((self.coupon_rate/12) * x) / (x - 1))
    
    
    #This function takes all the information we have on the mortgage and builds
    #a dataframe that has the periods, starting_balance, interest, principal, cashflow,
    #ending_balance, and effective coupon
    def cashflows(self):
        cashflow = []
        principal = self.outstanding_principal
        for period in range(self.term):
            interest_payment = principal * (self.coupon_rate / 12) #Calculate current interest payment
            principal_paid = self.payment - interest_payment #principal payment is amortized payment - interest payment
            
            #Input values into a temporary dictionary to be added to the cashflow list
            temp_dict = {'period': period + 1, 
                         'starting_balance': principal, 
                         'interest': interest_payment, 
                         'principal': principal_paid,
                         'cashflow': self.payment,
                         'ending_balance': principal - principal_paid,
                         'effective_coupon': self.coupon_rate * 100}
            
            cashflow.append(temp_dict)
            principal -= principal_paid
        return pd.DataFrame(cashflow).set_index('period') #return cashflow list as dataframe


#Define Bond Class
class Bond:
    #This class is initialized with a list of classes, list of principal balances, and a list of Mortgage objects.
    #A Bond can be made up of any number of classes as long as their are the same number of principal_balances
    #It is also required that the classes and principal_balances are input in the same order for correct naming.
    def __init__(self, classes: List[str], principal_balances: List[int], mortgages: List[Mortgage]):
        if len(classes) != len(principal_balances):
            raise ValueError("The lengths of 'classes' and 'principal_balances' must be the same.")
        
        self.classes = classes
        self.principal_balances = principal_balances
        self.mortgages = mortgages
        self.aggregated_cashflow = self.aggregate_cashflows() #A dataframe with all mortgage cashflows aggregated
        self.bond_df = self.waterfall() #A dataframe with the cashflow waterfall model implemented
        self.bond_WALs = self.weighted_average_life() #A dictionary containing the WAL for each bond class
        self.bond_total_cashflows = self.total_cashflows() #A dictionary containing the total cashflow for each bond class
        self.bond_IRRs = self.internal_rate_of_return() #A dictionary containing the IRR for each bond class
    
    
    #This function goes through each mortgage and adds all values by index which is period and by column
    def aggregate_cashflows(self):
        mortgage_sum = self.mortgages[0].cashflow_df

        for i in range(1, len(self.mortgages)):
            mortgage_sum = mortgage_sum.add(self.mortgages[i].cashflow_df, fill_value=0)
        
        #The effect coupon rate can then be calculated using the sum of all interest divided by the starting balance
        #This is then annualized and made to be represented as a percentage
        mortgage_sum['effective_coupon'] = mortgage_sum['interest']/mortgage_sum['starting_balance'] * 12 * 100
        
        return mortgage_sum
    
    
    #This function creates the waterfall model using the aggregated cashflow, classes, and principal_balances
    def waterfall(self):
        #start by creating a list of column names for each class in classes
        columns = ['period']
        for class_name in self.classes:
                columns.append(f'{class_name}_balance')
                columns.append( f'{class_name}_interest')
                columns.append(f'{class_name}_principal')
                columns.append(f'{class_name}_cashflow')
        cashflow_df = pd.DataFrame(columns=columns) #initialize empty dataframe with the column list
        
        #Iterate through the aggregated_cashflow
        principal_balances = self.principal_balances[:]
        for period, row in self.aggregated_cashflow.iterrows():
            interest_payment = row['interest'] #Current period total interest payment
            principal_payment = row['principal'] #Current period total principal payment
            starting_balance = row['starting_balance'] #Current period starting balance
            
            temp_dict = {'period': period}  #Iterate through each bond class
            for i in range(len(principal_balances)): 
                #If the current principal of the bond is greater than 0 we pay interest and principal
                if principal_balances[i] > 0:
                    #total interest paid for the current bond class
                    interest_paid = (principal_balances[i] / starting_balance) * interest_payment
                    #principal paid is the minimum of the current total principal payment and the principal balance
                    principal_paid = min(principal_payment, principal_balances[i]) 
                    principal_balances[i] -= principal_paid #pay principal to principal balance
                    #reduce principal payment by principal paid, if left over will be passed to next bond class
                    principal_payment -= principal_paid
                    #Add values to temporary dictionary until each class represented
                    temp_dict.update({
                        f'{self.classes[i]}_balance': principal_balances[i],
                        f'{self.classes[i]}_interest': interest_paid,
                        f'{self.classes[i]}_principal': principal_paid,
                        f'{self.classes[i]}_cashflow': principal_paid + interest_paid
                    })
                #Otherwise the bond class is fully paid off and nothing is paid
                else:
                    temp_dict.update({
                        'period': period,
                        f'{self.classes[i]}_balance': 0.0,
                        f'{self.classes[i]}_interest': 0.0,
                        f'{self.classes[i]}_principal': 0.0,
                        f'{self.classes[i]}_cashflow': 0.0
                    })
            #Add temporary dictionary from current period to cashflow_df
            cashflow_df =  pd.concat([cashflow_df, pd.DataFrame([temp_dict])], ignore_index=True)
        return cashflow_df
    
    
    #This function calculates the weighted average life for each bond using the classes and bond_df fields
    def weighted_average_life(self):
        wal = {}
        #Iterate through each class
        for i in range(len(self.classes)):
            #Get parts of bond_df where the current class's principal is greater than 0
            paid_period = self.bond_df.loc[self.bond_df[f'{self.classes[i]}_principal'] >= 0]
            wal.update({
                #We calculate the WAL using the sum of principal * (period/12) which is then divided by the starting balance
                f'Bond Class {self.classes[i]} WAL': round(sum(paid_period[f'{self.classes[i]}_principal'] * (paid_period['period']/12)) / self.principal_balances[i] , 2)
            })
        return wal
    
    
    #This function calculates the total cashflow for each bond using the classes and bond_df fields
    def total_cashflows(self):
        cashflows = {}
        #Iterate through each class
        for i in range(len(self.classes)):
            cashflows.update({
                #Sum all cashflow for the given bond class
                f'Bond Class {self.classes[i]} Total Cashflow': round(sum(self.bond_df[f'{self.classes[i]}_cashflow']), 2)
            })
        return cashflows
    
    
    #This function calculates the IRR for each bond using the classes and bond_df fields
    #This function also utilizes the numpy_finance library and the irr function within this library
    def internal_rate_of_return(self):
        irrs = {}
        #Iterate through each class
        for i in range(len(self.classes)):
            #Get list of cashflows with the first value being the negative starting principal of the bond class
            cashflow_list = np.insert(self.bond_df.loc[self.bond_df[f'{self.classes[i]}_cashflow'] != 0, f'{self.classes[i]}_cashflow'].values, 0, -(self.principal_balances[i]))
            irrs.update({
                #Pass irr function the cashflow_list
                #Annualize IRR by multiplying by 12 and multiply by 100 to get as a percent
                f'Bond Class {self.classes[i]} IRR': npf.irr(cashflow_list) * 12 * 100
            })
        return irrs
    

def main():
    st.title('Mortgage Backed Securities Cashflow Waterfall Model')
    
    # File Upload Section
    st.header('1. Load Loan Data')
    
    # Radio button to choose data source
    data_source = st.radio("Select Loan Data Source", 
                            ["Upload Excel", "Use Example Data"])
    
    if data_source  == "Use Example Data":

        loan_data = pd.read_excel("Simple Loan Tape.xlsx")

        st.subheader("Example Loan Portfolio")
        st.dataframe(loan_data)
    
    else:

        uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])

        if uploaded_file is None:
            st.info("Please upload an Excel file or select 'Use Example Data'")
            return
        
        loan_data = pd.read_excel(uploaded_file)

    required_columns = ['ID', 'Cut Off Date Balance', 'Gross Coupon', 'Remaining Amortization']
    if not all(col in loan_data.columns for col in required_columns):
        st.error(f"File must contain columns: {', '.join(required_columns)}")
        return

    #Portfolio summary
    total_balance = loan_data['Cut Off Date Balance'].sum()
    avg_coupon = loan_data['Gross Coupon'].mean()
    avg_term = loan_data['Remaining Amortization'].mean()
        
    st.subheader('Portfolio Summary')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Portfolio Balance", f"${total_balance:,.0f}")
    with col2:
        st.metric("Average Coupon Rate", f"{avg_coupon:.2f}%")
    with col3:
        st.metric("Average Remaining Term", f"{avg_term:.0f} months")
    


    st.header('2. Configure Bond Classes')

    total_balance = loan_data['Cut Off Date Balance'].sum()

    suggested_classes = ['Senior', 'Mezzanine', 'Junior']
    suggested_splits = [0.7, 0.2, 0.1]

    num_classes = st.number_input("Number of Bond Classes", min_value=1, max_value=5, value=3)

    # Dynamic class inputs
    classes = []
    principal_balances = []
    
    for i in range(num_classes):
        col1, col2 = st.columns(2)
        with col1:
            default_name = suggested_classes[i] if i < len(suggested_classes) else f'Class {chr(65+i)}'
            class_name = st.text_input(f'Class Name', value=default_name, key=f'class_name_{i}')
        
        with col2:
            default_split = suggested_splits[i] if i < len(suggested_splits) else 1/num_classes
            principal_balance = st.number_input(
                f'Principal Balance', 
                min_value=0.0, 
                max_value=float(total_balance), 
                value=total_balance * default_split,
                key=f'principal_balance_{i}'
            )
        
        classes.append(class_name)
        principal_balances.append(principal_balance)
    

    # Create Mortgages
    mortgages = []
    for _, row in loan_data.iterrows():
        mortgage_instance = Mortgage(
            outstanding_principal=row['Cut Off Date Balance'],
            coupon_rate=row['Gross Coupon']/100,
            term=row['Remaining Amortization']
        )
        mortgages.append(mortgage_instance)
    
    # Calculate Button
    if st.button('Calculate Waterfall'):
        # Validate principal balances
        if abs(sum(principal_balances) - total_balance) > 1:
            st.error("Total principal balances must equal total portfolio balance")
            return
        
        # Perform Calculation
        bonds = Bond(classes, principal_balances, mortgages)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            'Weighted Average Life', 
            'Total Cashflows', 
            'Internal Rate of Return', 
            'Waterfall Details'
        ])
        
        with tab1:
            st.header('Weighted Average Life (WAL)')
            st.dataframe(pd.DataFrame.from_dict(bonds.bond_WALs, orient='index', columns=['WAL (Years)']))
        
        with tab2:
            st.header('Total Cashflows')
            st.dataframe(pd.DataFrame.from_dict(bonds.bond_total_cashflows, orient='index', columns=['Total Cashflow']))
        
        with tab3:
            st.header('Internal Rate of Return (IRR)')
            st.dataframe(pd.DataFrame.from_dict(bonds.bond_IRRs, orient='index', columns=['IRR (%)']))
        
        with tab4:
            st.header('Waterfall Details')
            # Show first 20 periods of waterfall details
            st.dataframe(bonds.bond_df.head(20))

if __name__ == '__main__':
    main()