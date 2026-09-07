// In The Name of Allah
 
#include<bits/stdc++.h>
using namespace std;
 
#define ll long long
#define optimize()ios_base::sync_with_stdio(0);cin.tie(0);cout.tie(0);
#define pb push_back
#define all(v) v.begin(), v.end()
#define nl '\n'
#define mod 1000000007
#define ff first
#define ss second
#define pii pair<int ,int>
#define vi vector<int>
#define vpii vector<pii>
#define vll vector<ll>
#define lp(i, a, b) for (int i = a; i < b; ++i)
#define bitcnt(n) __builtin_popcountll(n)
#define bitclz(n) __builtin_clz(n)
#define bitprt(n) __builtin_parity(n)
#define bitff(n) __builtin_ffs(n)
       
void done(){
    int n; cin >> n;
    
    vector<int> a(n+1);
    
    for (int i=1; i<=n; i++){
        cin >> a[i];
    }
    
    if (a[n-1]>a[n]){
        cout << -1 << nl;
        return;
    }

    if (a[n]>=0){
        cout << n-2 << nl;
        for (int i=1; i<=n-2; i++){
            cout << i << " " << n-1 << " " << n << nl;
        }
    } 
    else {
        bool ok=true;
        for (int i=2; i<=n; i++){
            if (a[i-1]>a[i]){
                ok = false;
            }
        }
        if (ok){
            cout << 0 << nl;
        } 
        else {
            cout << -1 << nl;
        }
    }

}
     
int main() {
    optimize();
    int t; cin >> t;
    while (t--)done();
}