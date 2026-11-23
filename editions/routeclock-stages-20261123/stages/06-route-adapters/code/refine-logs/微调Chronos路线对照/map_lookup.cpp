// Expand causal route-map branches in the original insertion order.
// The caller retains NumPy's weighted mean and covariance reductions.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

struct Row { double x,y,z,support,weight,index; };
using Point = std::array<double,3>;
struct Edge { int to, count; double entry, bridge; Point dest; };
struct Map {
    const double *points, *arc, *tangent;
    const int *begin,*closed;
    std::vector<std::vector<Edge>> edges;
    std::vector<Row> rows;
    Point at(int f,double s,double &support) const {
        int lo=begin[f],hi=begin[f+1]-1;
        const double length=arc[hi];
        if (closed[f] && length>0) { s=std::fmod(s,length); if(s<0) s+=length; }
        int j=lo, tangent_at=lo; double dx=s;
        support=0;
        if(s>length){j=hi;tangent_at=hi-1;dx=s-length;}
        else if(s>=0){j=std::min(hi-1,std::max(lo,int(std::upper_bound(arc+lo,arc+hi+1,s)-arc)-1));tangent_at=j;dx=s-arc[j];support=1;}
        return {points[3*j]+dx*tangent[3*tangent_at],points[3*j+1]+dx*tangent[3*tangent_at+1],points[3*j+2]+dx*tangent[3*tangent_at+2]};
    }
    void expand(int f,double s,int depth,int query) {
        const int last=begin[f+1]-1;
        if(s>arc[last] && !closed[f] && !edges[f].empty() && depth<8){
            int total=0;for(const auto &e:edges[f])total+=e.count;
            for(const auto &e:edges[f]){
                const double rem=s-arc[last];
                if(e.bridge>1e-9 && rem<e.bridge){
                    Point p;
                    for(int k=0;k<3;k++)p[k]=points[3*last+k]+(e.dest[k]-points[3*last+k])*rem/e.bridge;
                    rows.push_back({p[0],p[1],p[2],1.,double(e.count)/total,double(query)});
                }else{
                    const size_t start=rows.size();expand(e.to,e.entry+rem-e.bridge,depth+1,query);
                    for(size_t j=start;j<rows.size();j++)rows[j].weight=rows[j].weight*e.count/total;
                }
            }
            return;
        }
        double supp;auto p=at(f,s,supp);rows.push_back({p[0],p[1],p[2],supp,1.,double(query)});
    }
};
extern "C" int lookup(int nf,const double *pts,const double *arc,const double *tan,const int *beg,const int *cl,
                      int ne,const int *src,const int *dst,const int *count,const double *entry,const double *bridge,const double *dest,
                      int nq,const int *fid,const double *coord,void **result) {
    try {
        Map map{pts,arc,tan,beg,cl,std::vector<std::vector<Edge>>(nf),{}};
        for(int i=0;i<ne;i++)map.edges[src[i]].push_back({dst[i],count[i],entry[i],bridge[i],{dest[3*i],dest[3*i+1],dest[3*i+2]}});
        map.rows.reserve(nq);
        for(int i=0;i<nq;i++)map.expand(fid[i],coord[i],0,i);
        *result=std::malloc(map.rows.size()*sizeof(Row));
        if(!*result)return -1;
        std::memcpy(*result,map.rows.data(),map.rows.size()*sizeof(Row));return int(map.rows.size());
    }catch(...){return -1;}
}
extern "C" void release(void *ptr){std::free(ptr);}
